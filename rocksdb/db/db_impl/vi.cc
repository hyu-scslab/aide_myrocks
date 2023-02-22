#ifdef VERSION_INDEX

#include "rocksdb/vi.h"

#include <fcntl.h>

#include "db/arena_wrapped_db_iter.h"
#include "db/dbformat.h"
#include "rocksdb/snapshot.h"
#include "rocksdb/types.h"
#include "table/block_based/block_based_table_iterator.h"

#ifdef VERSION_INDEX_DEBUG
#include <chrono>
#include <thread>

#include "rocksdb/sst_file_reader.h"
#endif

namespace ROCKSDB_NAMESPACE {

void ViPrescreen::Prepare(const std::string& sst_fname) {
  assert(!sst_fname.empty());
  assert(snapshot_ != nullptr);
  assert(snapshot_->GetSequenceNumber() > kMinUnCommittedSeq);
  assert(!prepared_);

  sst_fname_ = sst_fname;

  sst_fd_ = open(sst_fname.c_str(), O_RDONLY, (mode_t)0600);
  assert(sst_fd_ > 0);

  std::string vi_fname = sst_fname + ".vi";
  vi_fd_ = open(vi_fname.c_str(), O_RDONLY, (mode_t)0600);
  assert(vi_fd_ > 0);

  prepared_ = true;
}

void ViPrescreen::Run() {
  assert(prepared_);

  std::string i_fname = sst_fname_ + ".i";

  if (ShouldReuse()) {
    i_fd_ = open(i_fname.c_str(), O_RDONLY, (mode_t)0600);

    // Reusing the file means the file exists for this snapshot. If a new
    // transaction is held (i.e., new snapshot), we need to create a new one.
    assert(i_fd_ >= 0);

    uint32_t num_ctids = 0;
    read(i_fd_, reinterpret_cast<char*>(&num_ctids), sizeof(uint32_t));
    total_count_ += num_ctids;
    std::cout << "[ViPrescreen] (" << sst_fname_ << "): " << num_ctids
              << " (# indexes, REUSE)" << std::endl;

    return;
  }

  // Indicate that we can reuse the .i file created here if there's another call
  used_sst_.insert(sst_fname_);

  // Create the .i file
  i_fd_ = open(i_fname.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
  assert(i_fd_ > 0);

  uint32_t num_ctids = 0;
  int ret = 0;

  // The first 4 bytes are used to represent the number of CTIDs
  ret = write(i_fd_, reinterpret_cast<char*>(&num_ctids), sizeof(uint32_t));
  assert(ret == sizeof(uint32_t));

  ViCtidData vi_ctid[READ_CHUNK];
  char* buf = reinterpret_cast<char*>(vi_ctid);
  std::vector<CtidData> ctids;

  // Per SSTable prescreening using a vi file. The vi file is built per SSTable
  // and it contains a continuous array of ViCtidData.
  // TODO: optimization, currently reads ViCtidData one by one.
  bool data_remains = true;
  while (data_remains) {
    ret = read(vi_fd_, buf, sizeof(vi_ctid));
    uint32_t num_read = ret / sizeof(ViCtidData);

    for (uint32_t i = 0; i < num_read; ++i) {
      if (IsVisible(vi_ctid[i])) {
        ctids.push_back(vi_ctid[i].ctid);
      }
    }

    data_remains = (ret == sizeof(vi_ctid));
  }
  // We only close the .vi file fd since the other two files (i.e., .sst and .i)
  // are closed after offloading is done.
  close(vi_fd_);

  num_ctids = ctids.size();
  
  // Return -2 in the case of empty ctid file
  if (num_ctids == 0) {
    close(i_fd_);
    i_fd_ = VI_EMPTY_FD;
    std::cout << "[ViPrescreen] (" << sst_fname_ << "): " << num_ctids
              << " (# indexes, ZERO)" << std::endl;
    return;
  }

  for (auto& ctid : ctids) {
    write(i_fd_, reinterpret_cast<char*>(&ctid), sizeof(CtidData));
  }
  pwrite(i_fd_, reinterpret_cast<char*>(&num_ctids), sizeof(uint32_t), 0);

  std::cout << "[ViPrescreen] (" << sst_fname_ << "): " << num_ctids
            << " (# indexes)" << std::endl;

  ret = fsync(i_fd_);
  assert(ret == 0);

  total_count_ += num_ctids;
}

bool ViPrescreen::IsVisible(const ViCtidData& vi_ctid) {
  assert(snapshot_ != nullptr);
  const SequenceNumber seq = snapshot_->GetSequenceNumber();

  // For xmin, the value type is saved along with the sequence number.
  // We need to shift it to get the real xmin. xmax doesn't share the issue.
  uint64_t unpacked_xmin;
  ValueType type;

  uint64_t packed = DecodeFixed64((char*)(&vi_ctid.xmin));
  UnPackSequenceAndType(packed, &unpacked_xmin, &type);
  uint64_t unpacked_xmax = DecodeFixed64((char*)(&vi_ctid.xmax));

  bool xmin_in_range = (unpacked_xmin <= seq);
  bool xmax_in_range =
      (seq < unpacked_xmax) || (unpacked_xmax == kMinUnCommittedSeq);
  bool type_in_range = (type == ValueType::kTypeValue);

  return (xmin_in_range && xmax_in_range && type_in_range);
}

class ViDebugFormat {
 public:
  ViDebugFormat(Slice key, std::string sst_fname, uint32_t offset)
      : key_(key), sst_fname_(sst_fname), offset_(offset) {}

  friend auto operator<<(std::ostream& os, ViDebugFormat const& debug)
      -> std::ostream& {
    return os << debug.sst_fname_ << ", offset: " << debug.offset_ << " | "
              << ExtractXmin(debug.key_) << " ~ " << ExtractXmax(debug.key_);
  }

 private:
  Slice key_;
  std::string sst_fname_;
  uint32_t offset_;
};

#ifdef VERSION_INDEX_DEBUG_SYSBENCH
void ViPrescreen::RunDebug(const std::vector<std::string>& sst_list,
                           const Snapshot* snapshot) {
  assert(snapshot != nullptr);
  assert(snapshot->GetSequenceNumber() > kMinUnCommittedSeq);

  std::unordered_map<uint32_t, ViDebugFormat> result;

  for (auto& sst_fname : sst_list) {
    std::cout << "[ViPrescreenDebug] Start: " << sst_fname << std::endl;

    std::string vi_fname = sst_fname + ".vi";

    int sst_fd = open(sst_fname.c_str(), O_RDONLY, (mode_t)0600);
    assert(sst_fd > 0);
    int vi_fd = open(vi_fname.c_str(), O_RDONLY, (mode_t)0600);
    assert(vi_fd > 0);

    ViCtidData vi_ctid;
    std::vector<CtidData> ctids;

    uint32_t num_ctids = 0;
    uint32_t duplicate_key = 0;
    uint32_t inverse_key = 0;

    std::vector<uint32_t> user_keys;
    std::unordered_set<uint32_t> user_keys_set;

    while (read(vi_fd, reinterpret_cast<char*>(&vi_ctid), sizeof(ViCtidData)) !=
           0) {
      if (IsVisible(vi_ctid, snapshot->GetSequenceNumber())) {
        ctids.push_back(vi_ctid.ctid);

        char* buf = new char[28];
        pread(sst_fd, buf, 28, vi_ctid.ctid.offset);
        buf[4] &= 0x7F;
        uint32_t user_key = Decode4ByteKey(&buf[4]);
        Slice curr_key(buf, 28);
        SequenceNumber xmin = ExtractXmin(curr_key);
        SequenceNumber xmax = ExtractXmax(curr_key);
        ValueType xmin_type = ExtractValueType(curr_key);
        uint64_t packed_xmin = vi_ctid.xmin;
        uint64_t packed_xmax = vi_ctid.xmax;
        SequenceNumber unpacked_xmin;
        SequenceNumber unpacked_xmax;
        ValueType unpacked_xmin_type;
        UnPackSequenceAndType(packed_xmin, &unpacked_xmin, &unpacked_xmin_type);
        unpacked_xmax = DecodeFixed64((char*)(&packed_xmax));

        assert(xmin == unpacked_xmin);
        assert(xmax == unpacked_xmax);
        assert(xmin_type == ValueType::kTypeValue);
        assert(unpacked_xmin_type == xmin_type);

        ViDebugFormat debug(curr_key, sst_fname, vi_ctid.ctid.offset);
        auto iter = result.find(user_key);
        if (iter == result.end()) {
          result.insert(std::make_pair(user_key, debug));
        } else {
          ViDebugFormat prev_debug = iter->second;
          std::cout << "\tDuplicate key: " << user_key << std::endl
                    << "\t\t(prev): " << prev_debug << std::endl
                    << "\t\t(curr): " << debug << std::endl;
        }
      }
    }
    close(sst_fd);
    close(vi_fd);

    Options option;
    ReadOptions ropts;
    Status s;

    SstFileReader reader(option);
    s = reader.Open(sst_fname);
    assert(s.ok());
    s = reader.VerifyChecksum();
    assert(s.ok());

    ropts.snapshot = snapshot;
    ArenaWrappedDBIter* arena_iter =
        (ArenaWrappedDBIter*)reader.NewIterator(ropts);
    std::unique_ptr<Iterator> iter(arena_iter);
    BlockBasedTableIterator* table_iter =
        (BlockBasedTableIterator*)arena_iter->GetIterUnderDBIter();

    uint32_t num_visible_in_sst = 0;
    for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
      Slice key(table_iter->key());
      SequenceNumber xmin, xmax;
      ValueType type;
      uint64_t packed = ExtractInternalKeyFooter(key);

      UnPackSequenceAndType(packed, &xmin, &type);
      xmax = ExtractXmax(key);

      // The logic here must be the same as IsVisible()
      SequenceNumber seq = snapshot->GetSequenceNumber();
      bool xmin_in_range = (xmin <= seq);
      bool xmax_in_range = ((seq < xmax) || (xmax == kMinUnCommittedSeq));
      bool type_in_range = (type == ValueType::kTypeValue);

      if (xmin_in_range && xmax_in_range && type_in_range) {
        num_visible_in_sst++;
      }
    }

    assert(ctids.size() == num_visible_in_sst);

    num_ctids = ctids.size();

    std::cout << "[ViPrescreenDebug] (" << sst_fname << "): " << num_ctids
              << " (# indexes)" << std::endl;
  }
}

bool ViPrescreen::IsVisible(const ViCtidData& vi_ctid,
                            const SequenceNumber seq) {
  assert(seq > kMinUnCommittedSeq);

  // For xmin, the value type is saved along with the sequence number.
  // We need to shift it to get the real xmin. xmax doesn't share the issue.
  uint64_t unpacked_xmin;
  ValueType type;

  uint64_t packed = DecodeFixed64((char*)(&vi_ctid.xmin));
  UnPackSequenceAndType(packed, &unpacked_xmin, &type);
  uint64_t unpacked_xmax = DecodeFixed64((char*)(&vi_ctid.xmax));

  bool xmin_in_range = (unpacked_xmin <= seq);
  bool xmax_in_range =
      (seq < unpacked_xmax) || (unpacked_xmax == kMinUnCommittedSeq);
  bool type_in_range = (type == ValueType::kTypeValue);

  return (xmin_in_range && xmax_in_range && type_in_range);
}
#endif  // VERSION_INDEX_DEBUG_SYSBENCH

}  // namespace ROCKSDB_NAMESPACE

#endif  // VERSION_INDEX
