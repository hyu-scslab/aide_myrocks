//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// BlockBuilder generates blocks where keys are prefix-compressed:
//
// When we store a key, we drop the prefix shared with the previous
// string.  This helps reduce the space requirement significantly.
// Furthermore, once every K keys, we do not apply the prefix
// compression and store the entire key.  We call this a "restart
// point".  The tail end of the block stores the offsets of all of the
// restart points, and can be used to do a binary search when looking
// for a particular key.  Values are stored as-is (without compression)
// immediately following the corresponding key.
//
// An entry for a particular key-value pair has the form:
//     shared_bytes: varint32
//     unshared_bytes: varint32
//     value_length: varint32
//     key_delta: char[unshared_bytes]
//     value: char[value_length]
// shared_bytes == 0 for restart points.
//
// The trailer of the block has the form:
//     restarts: uint32[num_restarts]
//     num_restarts: uint32
// restarts[i] contains the offset within the block of the ith restart point.

#include "table/block_based/block_builder.h"

#include <assert.h>
#include <algorithm>
#include "db/dbformat.h"
#include "rocksdb/comparator.h"
#include "table/block_based/data_block_footer.h"
#include "util/coding.h"
#ifdef VERSION_INDEX
#include "rocksdb/vi.h"
#endif

namespace ROCKSDB_NAMESPACE {

BlockBuilder::BlockBuilder(
    int block_restart_interval, bool use_delta_encoding,
    bool use_value_delta_encoding,
    BlockBasedTableOptions::DataBlockIndexType index_type,
    double data_block_hash_table_util_ratio)
    : block_restart_interval_(block_restart_interval),
      use_delta_encoding_(use_delta_encoding),
      use_value_delta_encoding_(use_value_delta_encoding),
      restarts_(1, 0),  // First restart point is at offset 0
      counter_(0),
      finished_(false) {
  switch (index_type) {
    case BlockBasedTableOptions::kDataBlockBinarySearch:
      break;
    case BlockBasedTableOptions::kDataBlockBinaryAndHash:
      data_block_hash_index_builder_.Initialize(
          data_block_hash_table_util_ratio);
      break;
    default:
      assert(0);
  }
  assert(block_restart_interval_ >= 1);
  estimate_ = sizeof(uint32_t) + sizeof(uint32_t);
}

void BlockBuilder::Reset() {
  buffer_.clear();
  restarts_.resize(1);  // First restart point is at offset 0
  assert(restarts_[0] == 0);
  estimate_ = sizeof(uint32_t) + sizeof(uint32_t);
  counter_ = 0;
  finished_ = false;
  last_key_.clear();
  if (data_block_hash_index_builder_.Valid()) {
    data_block_hash_index_builder_.Reset();
  }
#ifndef NDEBUG
  add_with_last_key_called_ = false;
#endif
}

void BlockBuilder::SwapAndReset(std::string& buffer) {
  std::swap(buffer_, buffer);
  Reset();
}

#ifdef VERSION_INDEX
void BlockBuilder::ViReset() {
  vi_buffer_.clear();
}

void BlockBuilder::ViSwapAndReset(std::string& buffer) {
  std::swap(vi_buffer_, buffer);
  ViReset();
}


size_t BlockBuilder::EstimateSizeOfVersionIndex() const {
  return vi_buffer_.size();
}
#endif

size_t BlockBuilder::EstimateSizeAfterKV(const Slice& key,
                                         const Slice& value) const {
  size_t estimate = CurrentSizeEstimate();
  // Note: this is an imprecise estimate as it accounts for the whole key size
  // instead of non-shared key size.
  estimate += key.size();
  // In value delta encoding we estimate the value delta size as half the full
  // value size since only the size field of block handle is encoded.
  estimate +=
      !use_value_delta_encoding_ || (counter_ >= block_restart_interval_)
          ? value.size()
          : value.size() / 2;

  if (counter_ >= block_restart_interval_) {
    estimate += sizeof(uint32_t);  // a new restart entry.
  }

  estimate += sizeof(int32_t);  // varint for shared prefix length.
  // Note: this is an imprecise estimate as we will have to encoded size, one
  // for shared key and one for non-shared key.
  estimate += VarintLength(key.size());  // varint for key length.
  if (!use_value_delta_encoding_ || (counter_ >= block_restart_interval_)) {
    estimate += VarintLength(value.size());  // varint for value length.
  }

  return estimate;
}

Slice BlockBuilder::Finish() {
  // Append restart array
  for (size_t i = 0; i < restarts_.size(); i++) {
    PutFixed32(&buffer_, restarts_[i]);
  }

  uint32_t num_restarts = static_cast<uint32_t>(restarts_.size());
  BlockBasedTableOptions::DataBlockIndexType index_type =
      BlockBasedTableOptions::kDataBlockBinarySearch;
  if (data_block_hash_index_builder_.Valid() &&
      CurrentSizeEstimate() <= kMaxBlockSizeSupportedByHashIndex) {
    data_block_hash_index_builder_.Finish(buffer_);
    index_type = BlockBasedTableOptions::kDataBlockBinaryAndHash;
  }

  // footer is a packed format of data_block_index_type and num_restarts
  uint32_t block_footer = PackIndexTypeAndNumRestarts(index_type, num_restarts);

  PutFixed32(&buffer_, block_footer);
  finished_ = true;
  return Slice(buffer_);
}

#ifdef VERSION_INDEX
void BlockBuilder::Add(const Slice& key, const Slice& value,
                       const Slice* const delta_value,
                       const uint64_t offset,
                       const uint32_t vi_offset,
                       const int nullbytes) {
#else
void BlockBuilder::Add(const Slice& key, const Slice& value,
                       const Slice* const delta_value) {
#endif
  // Ensure no unsafe mixing of Add and AddWithLastKey
  assert(!add_with_last_key_called_);

#ifdef VERSION_INDEX
  AddWithLastKeyImpl(key, value, last_key_, delta_value, buffer_.size(),
                     offset, vi_offset, nullbytes);
#else
  AddWithLastKeyImpl(key, value, last_key_, delta_value, buffer_.size());
#endif
  if (use_delta_encoding_) {
    // Update state
    // We used to just copy the changed data, but it appears to be
    // faster to just copy the whole thing.
    last_key_.assign(key.data(), key.size());
  }
}

#ifdef VERSION_INDEX
void BlockBuilder::AddWithLastKey(const Slice& key, const Slice& value,
                                  const Slice& last_key_param,
                                  const Slice* const delta_value,
                                  const uint64_t offset,
                                  const uint32_t vi_offset,
                                  const int nullbytes) {
#else
void BlockBuilder::AddWithLastKey(const Slice& key, const Slice& value,
                                  const Slice& last_key_param,
                                  const Slice* const delta_value) {
#endif
  // Ensure no unsafe mixing of Add and AddWithLastKey
  assert(last_key_.empty());
#ifndef NDEBUG
  add_with_last_key_called_ = false;
#endif

  // Here we make sure to use an empty `last_key` on first call after creation
  // or Reset. This is more convenient for the caller and we can be more
  // clever inside BlockBuilder. On this hot code path, we want to avoid
  // conditional jumps like `buffer_.empty() ? ... : ...` so we can use a
  // fast min operation instead, with an assertion to be sure our logic is
  // sound.
  size_t buffer_size = buffer_.size();
  size_t last_key_size = last_key_param.size();
  assert(buffer_size == 0 || buffer_size >= last_key_size);

  Slice last_key(last_key_param.data(), std::min(buffer_size, last_key_size));

#ifdef VERSION_INDEX
  AddWithLastKeyImpl(key, value, last_key, delta_value, buffer_size,
                     offset, vi_offset, nullbytes);
#else
  AddWithLastKeyImpl(key, value, last_key, delta_value, buffer_size);
#endif
}

#ifdef VERSION_INDEX
inline void BlockBuilder::AddWithLastKeyImpl(const Slice& key,
                                             const Slice& value,
                                             const Slice& last_key,
                                             const Slice* const delta_value,
                                             size_t buffer_size,
                                             const uint64_t offset,
                                             const uint32_t vi_offset,
                                             const int nullbytes) {
#else
inline void BlockBuilder::AddWithLastKeyImpl(const Slice& key,
                                             const Slice& value,
                                             const Slice& last_key,
                                             const Slice* const delta_value,
                                             size_t buffer_size) {
#endif
  assert(!finished_);
  assert(counter_ <= block_restart_interval_);
  assert(!use_value_delta_encoding_ || delta_value);
  size_t shared = 0;  // number of bytes shared with prev key
  if (counter_ >= block_restart_interval_) {
    // Restart compression
    restarts_.push_back(static_cast<uint32_t>(buffer_size));
    estimate_ += sizeof(uint32_t);
    counter_ = 0;
  } else if (use_delta_encoding_) {
    // See how much sharing to do with previous string
    shared = key.difference_offset(last_key);
  }

  const size_t non_shared = key.size() - shared;

  if (use_value_delta_encoding_) {
    // Add "<shared><non_shared>" to buffer_
    PutVarint32Varint32(&buffer_, static_cast<uint32_t>(shared),
                        static_cast<uint32_t>(non_shared));
  } else {
    // Add "<shared><non_shared><value_size>" to buffer_
    PutVarint32Varint32Varint32(&buffer_, static_cast<uint32_t>(shared),
                                static_cast<uint32_t>(non_shared),
                                static_cast<uint32_t>(value.size()));
  }
#ifdef VERSION_INDEX
  uint32_t record_offset;
  uint32_t key_offset;
  uint32_t header_size;
  uint32_t record_size;

  if (!is_system) {
    const uint32_t back_ptr = vi_offset + vi_buffer_.size();

    // Last 4 byte = back pointer.
    // This line must come before we copy the key data into the buffer_.
    char *back_ptr_address = const_cast<char*>(key.data() + key.size()
                                               - kNumInternalBytes + 16);
    EncodeFixed32(back_ptr_address, back_ptr);

    // Format of RocksDB K/V w\o delta encoding, and it's corresponding info
    // about CTID.
    //
    //                 off                off + h_len
    //                  v                     v
    // +----------------+-------+-------------+---------+
    // |  Varint32 ...  |  Key  |  nullbytes  |  Value  |
    // +----------------+-------+-------------+---------+
    //                  |                     |         |
    //                  +------- h_len -------+         |
    //                  |                               |
    //                  +------------- len -------------+
    key_offset = buffer_.size();

    // The buffer contains the actual raw data which will be put into the SST
    // file on flush. Thus, the offset may be calculated by the current file
    // offset + the buffer's current size.
    record_offset = offset + key_offset;
  }
#endif

  // Add string delta to buffer_ followed by value
  buffer_.append(key.data() + shared, non_shared);

#ifdef VERSION_INDEX
  if (!is_system) {
    // TODO: Delete this. No more use but left out for historical reason.
    // After key insertion
    //
    //           key_offset   buffer_.size()
    //                  v       v
    // +----------------+-------+--------------------------------+-----
    // |  Varint32 ...  |  Key  |  nullbytes (encoded in value)  | ...
    // +----------------+-------+--------------------------------+-----
    // header_size = buffer_.size() - key_offset + nullbytes;

    //
    // JAECHAN: 2022.11.03
    //
    // We point the front of the key which makes header size 0, but the first
    // 4 byte is used as internal key as mentioned in the MyRocks paper and
    // wiki. Thus, we set the header size into 4 byte so that the
    // offset + header size now points to the starting position of the user key.
    //
    // https://github.com/facebook/mysql-5.6/wiki/MyRocks-record-format#primary-key
    header_size = 4;
  }
#endif

  // Use value delta encoding only when the key has shared bytes. This would
  // simplify the decoding, where it can figure which decoding to use simply by
  // looking at the shared bytes size.
  if (shared != 0 && use_value_delta_encoding_) {
    buffer_.append(delta_value->data(), delta_value->size());
  } else {
    buffer_.append(value.data(), value.size());
  }
#ifdef VERSION_INDEX
  if (!is_system) {
    // After value insertion
    //
    //             key_offset                                buffer_.size()
    //                  v                                          v
    // +----------------+-------+-------------+----------+---------+
    // |  Varint32 ...  |  Key  |  nullbytes (in value)  |  Value  |
    // +----------------+-------+-------------+----------+---------+
    record_size = buffer_.size() - key_offset;
  }
#endif

  if (data_block_hash_index_builder_.Valid()) {
    data_block_hash_index_builder_.Add(ExtractUserKey(key),
                                       restarts_.size() - 1);
  }

  counter_++;
  estimate_ += buffer_.size() - buffer_size;

#ifdef VERSION_INDEX
  if (!is_system) {
    ViCtidData vi_ctid;

    // Decode xmin/xmax from key slice.
    vi_ctid.xmin = ExtractSeq(key);
    vi_ctid.xmax = ExtractXmax(key);

    vi_ctid.ctid.offset = record_offset;
    vi_ctid.ctid.len = record_size;
    vi_ctid.ctid.h_len = header_size;

    // Null bitmap is packed inside the value byte array, at the front.
    uint32_t bitpack = 0;
    if (nullbytes > 0) {
      memcpy((char*) &bitpack, value.data(), nullbytes);

      // In MyRocks, 0 indicates that the attribute IS NOT null and
      // 1 indicates that the field IS null. See m_null_mask in rdb_datadic.h.
      // We have to make 0 indicate that the attribute IS null and
      // 1 to indicate that the attribute IS NOT null.
      bitpack = (~bitpack) & 0x7FFFFFU;
    } else {
      // All filled with 1 to indicate that every attribute exists.
      bitpack = 0x7FFFFFU;
    }
    vi_ctid.ctid.bits = bitpack;

    vi_buffer_.append((char*) &vi_ctid, sizeof(ViCtidData));
  }
#endif
}

}  // namespace ROCKSDB_NAMESPACE
