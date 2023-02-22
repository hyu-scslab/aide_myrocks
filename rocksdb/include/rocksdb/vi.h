#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>

#include "port/port_posix.h"
#include "rocksdb/types.h"

namespace ROCKSDB_NAMESPACE {

struct __attribute__((__packed__)) CtidData {
  unsigned offset : 30, len : 10, h_len : 9, bits : 23;
};

/*
 * Version index structure.
 * CTID = Raw CTID + (xmin, xmax)
 *
 * +--------------------------------+
 * | xmin, xmax (64 + 64 bit)       |
 * +--------------------------------+--\
 * | in-segment offset (30 bit)     |   \
 * +--------------------------------+    \
 * | version length (10 bit)        |    |
 * +--------------------------------+   CTID (Canonical Tuple IDentifier)
 * | version header length (16 bit) |    |
 * +--------------------------------+    /
 * | NULL bitmap (23 bit)           |   /
 * +--------------------------------+--/
 */
struct ViCtidData {
  SequenceNumber xmin;
  SequenceNumber xmax;

  CtidData ctid;

  friend auto operator<<(std::ostream &os, ViCtidData const &vi_ctid)
      -> std::ostream & {
    return os << vi_ctid.xmin << ", " << vi_ctid.xmax << " | "
              << vi_ctid.ctid.offset << ", " << vi_ctid.ctid.len << ", "
              << vi_ctid.ctid.h_len << ", " << vi_ctid.ctid.bits;
  }
};

// The lifetime of a VersionIndexer is a single offload of a relation.
// Within this class, we also have variables for each 'run', which represents
// a round of prescreening for a single SSTable.
class ViPrescreen {
 public:
  ViPrescreen(const Snapshot *snapshot)
      : snapshot_(snapshot),
        total_count_(0),
        prepared_(false),
        sst_fd_(-1),
        vi_fd_(-1),
        i_fd_(-1) {}

  // Every execution of prescreening a SSTable should contain the cycle of
  // Prepare() -> Run() -> Reset().

  // Checks for basic issues (e.g., file not exist) and open up the necessary
  // files. We set the prepared flag to indicate that all preparation is ready.
  void Prepare(const std::string &sst_fname);

  // Prescreens a SSTable's version index file (e.g., 000...00035.sst.vi) and
  // creates a per SSTable index file (e.g., 000...00035.sst.i). Returns a
  // file descriptor with the index file opened. The caller is responsible
  // for closing the file descriptor after use.
  void Run();

  // Clears out the current Run() material.
  void Reset() {
    prepared_ = false;
    sst_fname_.clear();
    sst_fd_ = -1;
    vi_fd_ = -1;
    i_fd_ = -1;
  }

  void ResetTotalCount() {
    total_count_ = 0;
  }

  int SstFile() { return sst_fd_; }

  int IndexFile() { return i_fd_; }

  uint32_t TotalCount() { return total_count_; }

#ifdef VERSION_INDEX_DEBUG_SYSBENCH
  // Stand-alone version for debugging.
  static void RunDebug(const std::vector<std::string> &sst_list,
                       const Snapshot *snapshot);
#endif  // VERSION_INDEX_DEBUG_SYSBENCH

  const static int VI_EMPTY_FD = -2;

 private:
  bool ShouldReuse() { return (used_sst_.find(sst_fname_) != used_sst_.end()); }

  bool IsVisible(const ViCtidData &vi_ctid);

#ifdef VERSION_INDEX_DEBUG_SYSBENCH
  // Stand-alone version used in RunDebug().
  static bool IsVisible(const ViCtidData &vi_ctid, const SequenceNumber seq);
#endif  // VERSION_INDEX_DEBUG_SYSBENCH

  const Snapshot *snapshot_;
  std::unordered_set<std::string> used_sst_;
  uint32_t total_count_;

  const static uint32_t READ_CHUNK = 8192 / sizeof(ViCtidData);

  // Current use from Prepare() ~ Run()
  // These must be reset in every loop by Reset()
  bool prepared_;
  int sst_fd_;  // .sst file descriptor
  int vi_fd_;   // .vi  file descriptor
  int i_fd_;    // .i   file descriptor
  std::string sst_fname_;
};

inline bool IsUserTable(uint32_t cf_id) { return (cf_id > 1); }

inline uint32_t Decode4ByteKey(const char *buf) {
  return ((static_cast<uint32_t>(static_cast<unsigned char>(buf[3])) |
           (static_cast<uint32_t>(static_cast<unsigned char>(buf[2])) << 8) |
           (static_cast<uint32_t>(static_cast<unsigned char>(buf[1])) << 16) |
           (static_cast<uint32_t>(static_cast<unsigned char>(buf[0])) << 24))) &
         0x7FFFFFFF;
}

class ViUtil {
 public:
  static void Close(int fd) { close(fd); }
};

/******************************************************************************
 * Features related to tracking xmax writes. We track both the flush dependency
 * and the compaction dependency within a global dependency graph. Also, the
 * updaters should use ViContext to mark where the previous version is. That
 * way, we can write out the xmax value during transaction's commit phase.
 ******************************************************************************/
struct CompactionDependency {
  std::atomic<bool> is_done{false};
  uint32_t cnt = 0;
  std::unordered_set<uint64_t> outputs;
};

// MemTable < memtable id, sst number >
inline std::unordered_map<uint64_t, CompactionDependency *> memtable_dependency;
inline port::Mutex memtable_dependency_mutex;

inline std::unordered_set<uint64_t> live_memtables;
inline port::Mutex live_memtables_mutex;

// SST < sst number, sst number(s) >
inline std::unordered_map<uint64_t, CompactionDependency *> sst_dependency;
inline port::Mutex sst_dependency_mutex;

class ViContext {
 public:
  enum class ViOperationType : uint8_t { NONE = 0, UPDATE, DELETE };

  enum class ViTargetLocation : uint8_t { NONE = 0, MEMTABLE, SST };

  ViOperationType optype;  // Operation type for distinguishing normal Get()
  ViTargetLocation loc;    // MemTable or SST file
  std::string fname;       // SST file name
  uint32_t off;            // xmax offset within SST
  uint64_t mem_id;         // MemTable ID
  char *address;           // xmax offset within MemTable
  Slice key;               // the ENTIRE key (with seq, xmax, vi offset)
  char *buf;
  uint64_t cf_id;

  ViContext() { Reset(); }

  ViContext(uint64_t _cf_id) {
    Reset();
    cf_id = _cf_id;
  }

  ~ViContext() {
    if (buf != nullptr) {
      delete[] buf;
    }
  }

  void Reset() {
    optype = ViOperationType::NONE;
    loc = ViTargetLocation::NONE;
    fname.clear();
    off = 0;
    mem_id = 0;
    address = nullptr;
    buf = nullptr;
    cf_id = 0;
  }

  bool IsNone() { return (loc == ViTargetLocation::NONE); }

  bool IsUpdate() { return (optype == ViOperationType::UPDATE); }

  // Should be the same as IsUserTable()
  bool IsUserTable() { return (cf_id > 1); }

  // Old version in memtable must be marked by using this function.
  void MarkMemTable(const Slice &k, uint64_t id, char *addr) {
    assert(IsNone());
    loc = ViTargetLocation::MEMTABLE;
    mem_id = id;
    address = addr;
    CopyKey(k);
    key = Slice(buf, k.size());
  }

  // Old version in SSTable must be marked by using this function.
  void MarkSST(const Slice &k, const std::string &file, uint32_t offset) {
    loc = ViTargetLocation::SST;
    fname = file;
    off = offset;
    CopyKey(k);
    key = Slice(buf, k.size());
  }

  void CopyKey(const Slice &k) {
    // The buffer allocated here is deleted during the commit phase.
    buf = new char[k.size()];
    memcpy(buf, k.data(), k.size());
  }

  friend auto operator<<(std::ostream &os, ViContext const &ctx)
      -> std::ostream & {
    if (ctx.loc == ViTargetLocation::MEMTABLE) {
      return os << "MEMTABLE | " << ctx.mem_id << ", " << (void *)ctx.address;
    } else if (ctx.loc == ViTargetLocation::SST) {
      return os << "SST | " << ctx.fname << ", " << ctx.off;
    } else {
      return os << "NONE | unknown";
    }
  }
};

inline thread_local ViContext *vi_ctx = nullptr;

}  // namespace ROCKSDB_NAMESPACE
