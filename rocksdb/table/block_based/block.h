//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#pragma once
#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>
#include <sys/types.h>
#include <pwd.h>

#include "db/pinned_iterators_manager.h"
#include "port/malloc.h"
#include "rocksdb/iterator.h"
#include "rocksdb/options.h"
#include "rocksdb/statistics.h"
#include "rocksdb/table.h"
#include "table/block_based/block_prefix_index.h"
#include "table/block_based/data_block_hash_index.h"
#include "table/format.h"
#include "table/internal_iterator.h"
#include "test_util/sync_point.h"
#include "util/random.h"

#ifdef SMARTSSD
#include "ndp_functions.h"
#endif //SMARTSSD

namespace ROCKSDB_NAMESPACE {

struct BlockContents;
class Comparator;
template <class TValue>
class BlockIter;
class DataBlockIter;
class IndexBlockIter;
class MetaBlockIter;
class BlockPrefixIndex;

// BlockReadAmpBitmap is a bitmap that map the ROCKSDB_NAMESPACE::Block data
// bytes to a bitmap with ratio bytes_per_bit. Whenever we access a range of
// bytes in the Block we update the bitmap and increment
// READ_AMP_ESTIMATE_USEFUL_BYTES.
class BlockReadAmpBitmap {
 public:
  explicit BlockReadAmpBitmap(size_t block_size, size_t bytes_per_bit,
                              Statistics* statistics)
      : bitmap_(nullptr),
        bytes_per_bit_pow_(0),
        statistics_(statistics),
        rnd_(Random::GetTLSInstance()->Uniform(
            static_cast<int>(bytes_per_bit))) {
    TEST_SYNC_POINT_CALLBACK("BlockReadAmpBitmap:rnd", &rnd_);
    assert(block_size > 0 && bytes_per_bit > 0);

    // convert bytes_per_bit to be a power of 2
    while (bytes_per_bit >>= 1) {
      bytes_per_bit_pow_++;
    }

    // num_bits_needed = ceil(block_size / bytes_per_bit)
    size_t num_bits_needed = ((block_size - 1) >> bytes_per_bit_pow_) + 1;
    assert(num_bits_needed > 0);

    // bitmap_size = ceil(num_bits_needed / kBitsPerEntry)
    size_t bitmap_size = (num_bits_needed - 1) / kBitsPerEntry + 1;

    // Create bitmap and set all the bits to 0
    bitmap_ = new std::atomic<uint32_t>[bitmap_size]();

    RecordTick(GetStatistics(), READ_AMP_TOTAL_READ_BYTES, block_size);
  }

  ~BlockReadAmpBitmap() { delete[] bitmap_; }

  void Mark(uint32_t start_offset, uint32_t end_offset) {
    assert(end_offset >= start_offset);
    // Index of first bit in mask
    uint32_t start_bit =
        (start_offset + (1 << bytes_per_bit_pow_) - rnd_ - 1) >>
        bytes_per_bit_pow_;
    // Index of last bit in mask + 1
    uint32_t exclusive_end_bit =
        (end_offset + (1 << bytes_per_bit_pow_) - rnd_) >> bytes_per_bit_pow_;
    if (start_bit >= exclusive_end_bit) {
      return;
    }
    assert(exclusive_end_bit > 0);

    if (GetAndSet(start_bit) == 0) {
      uint32_t new_useful_bytes = (exclusive_end_bit - start_bit)
                                  << bytes_per_bit_pow_;
      RecordTick(GetStatistics(), READ_AMP_ESTIMATE_USEFUL_BYTES,
                 new_useful_bytes);
    }
  }

  Statistics* GetStatistics() {
    return statistics_.load(std::memory_order_relaxed);
  }

  void SetStatistics(Statistics* stats) { statistics_.store(stats); }

  uint32_t GetBytesPerBit() { return 1 << bytes_per_bit_pow_; }

  size_t ApproximateMemoryUsage() const {
#ifdef ROCKSDB_MALLOC_USABLE_SIZE
    return malloc_usable_size((void*)this);
#endif  // ROCKSDB_MALLOC_USABLE_SIZE
    return sizeof(*this);
  }

 private:
  // Get the current value of bit at `bit_idx` and set it to 1
  inline bool GetAndSet(uint32_t bit_idx) {
    const uint32_t byte_idx = bit_idx / kBitsPerEntry;
    const uint32_t bit_mask = 1 << (bit_idx % kBitsPerEntry);

    return bitmap_[byte_idx].fetch_or(bit_mask, std::memory_order_relaxed) &
           bit_mask;
  }

  const uint32_t kBytesPersEntry = sizeof(uint32_t);   // 4 bytes
  const uint32_t kBitsPerEntry = kBytesPersEntry * 8;  // 32 bits

  // Bitmap used to record the bytes that we read, use atomic to protect
  // against multiple threads updating the same bit
  std::atomic<uint32_t>* bitmap_;
  // (1 << bytes_per_bit_pow_) is bytes_per_bit. Use power of 2 to optimize
  // muliplication and division
  uint8_t bytes_per_bit_pow_;
  // Pointer to DB Statistics object, Since this bitmap may outlive the DB
  // this pointer maybe invalid, but the DB will update it to a valid pointer
  // by using SetStatistics() before calling Mark()
  std::atomic<Statistics*> statistics_;
  uint32_t rnd_;
};

// This Block class is not for any old block: it is designed to hold only
// uncompressed blocks containing sorted key-value pairs. It is thus
// suitable for storing uncompressed data blocks, index blocks (including
// partitions), range deletion blocks, properties blocks, metaindex blocks,
// as well as the top level of the partitioned filter structure (which is
// actually an index of the filter partitions). It is NOT suitable for
// compressed blocks in general, filter blocks/partitions, or compression
// dictionaries (since the latter do not contain sorted key-value pairs).
// Use BlockContents directly for those.
//
// See https://github.com/facebook/rocksdb/wiki/Rocksdb-BlockBasedTable-Format
// for details of the format and the various block types.
class Block {
 public:
  // Initialize the block with the specified contents.
  explicit Block(BlockContents&& contents, size_t read_amp_bytes_per_bit = 0,
                 Statistics* statistics = nullptr);
  // No copying allowed
  Block(const Block&) = delete;
  void operator=(const Block&) = delete;

  ~Block();

  size_t size() const { return size_; }
  const char* data() const { return data_; }
  // The additional memory space taken by the block data.
  size_t usable_size() const { return contents_.usable_size(); }
  uint32_t NumRestarts() const;
  bool own_bytes() const { return contents_.own_bytes(); }

  BlockBasedTableOptions::DataBlockIndexType IndexType() const;

  // raw_ucmp is a raw (i.e., not wrapped by `UserComparatorWrapper`) user key
  // comparator.
  //
  // If iter is null, return new Iterator
  // If iter is not null, update this one and return it as Iterator*
  //
  // Updates read_amp_bitmap_ if it is not nullptr.
  //
  // If `block_contents_pinned` is true, the caller will guarantee that when
  // the cleanup functions are transferred from the iterator to other
  // classes, e.g. PinnableSlice, the pointer to the bytes will still be
  // valid. Either the iterator holds cache handle or ownership of some resource
  // and release them in a release function, or caller is sure that the data
  // will not go away (for example, it's from mmapped file which will not be
  // closed).
  //
  // NOTE: for the hash based lookup, if a key prefix doesn't match any key,
  // the iterator will simply be set as "invalid", rather than returning
  // the key that is just pass the target key.
  DataBlockIter* NewDataIterator(const Comparator* raw_ucmp,
                                 SequenceNumber global_seqno,
                                 DataBlockIter* iter = nullptr,
                                 Statistics* stats = nullptr,
                                 bool block_contents_pinned = false);

  // Returns an MetaBlockIter for iterating over blocks containing metadata
  // (like Properties blocks).  Unlike data blocks, the keys for these blocks
  // do not contain sequence numbers, do not use a user-define comparator, and
  // do not track read amplification/statistics.  Additionally, MetaBlocks will
  // not assert if the block is formatted improperly.
  //
  // If `block_contents_pinned` is true, the caller will guarantee that when
  // the cleanup functions are transferred from the iterator to other
  // classes, e.g. PinnableSlice, the pointer to the bytes will still be
  // valid. Either the iterator holds cache handle or ownership of some resource
  // and release them in a release function, or caller is sure that the data
  // will not go away (for example, it's from mmapped file which will not be
  // closed).
  MetaBlockIter* NewMetaIterator(bool block_contents_pinned = false);

  // raw_ucmp is a raw (i.e., not wrapped by `UserComparatorWrapper`) user key
  // comparator.
  //
  // key_includes_seq, default true, means that the keys are in internal key
  // format.
  // value_is_full, default true, means that no delta encoding is
  // applied to values.
  //
  // If `prefix_index` is not nullptr this block will do hash lookup for the key
  // prefix. If total_order_seek is true, prefix_index_ is ignored.
  //
  // `have_first_key` controls whether IndexValue will contain
  // first_internal_key. It affects data serialization format, so the same value
  // have_first_key must be used when writing and reading index.
  // It is determined by IndexType property of the table.
  IndexBlockIter* NewIndexIterator(const Comparator* raw_ucmp,
                                   SequenceNumber global_seqno,
                                   IndexBlockIter* iter, Statistics* stats,
                                   bool total_order_seek, bool have_first_key,
                                   bool key_includes_seq, bool value_is_full,
                                   bool block_contents_pinned = false,
                                   BlockPrefixIndex* prefix_index = nullptr);

  // Report an approximation of how much memory has been used.
  size_t ApproximateMemoryUsage() const;

 private:
  BlockContents contents_;
  const char* data_;         // contents_.data.data()
  size_t size_;              // contents_.data.size()
  uint32_t restart_offset_;  // Offset in data_ of restart array
  uint32_t num_restarts_;
  std::unique_ptr<BlockReadAmpBitmap> read_amp_bitmap_;
  DataBlockHashIndex data_block_hash_index_;
};

// A `BlockIter` iterates over the entries in a `Block`'s data buffer. The
// format of this data buffer is an uncompressed, sorted sequence of key-value
// pairs (see `Block` API for more details).
//
// Notably, the keys may either be in internal key format or user key format.
// Subclasses are responsible for configuring the key format.
//
// `BlockIter` intends to provide final overrides for all of
// `InternalIteratorBase` functions that can move the iterator. It does
// this to guarantee `UpdateKey()` is called exactly once after each key
// movement potentially visible to users. In this step, the key is prepared
// (e.g., serialized if global seqno is in effect) so it can be returned
// immediately when the user asks for it via calling `key() const`.
//
// For its subclasses, it provides protected variants of the above-mentioned
// final-overridden methods. They are named with the "Impl" suffix, e.g.,
// `Seek()` logic would be implemented by subclasses in `SeekImpl()`. These
// "Impl" functions are responsible for positioning `raw_key_` but not
// invoking `UpdateKey()`.
template <class TValue>
class BlockIter : public InternalIteratorBase<TValue> {
 public:
  // Makes Valid() return false, status() return `s`, and Seek()/Prev()/etc do
  // nothing. Calls cleanup functions.
  virtual void Invalidate(const Status& s) {
    // Assert that the BlockIter is never deleted while Pinning is Enabled.
    assert(!pinned_iters_mgr_ || !pinned_iters_mgr_->PinningEnabled());

    data_ = nullptr;
    current_ = restarts_;
    status_ = s;

    // Call cleanup callbacks.
    Cleanable::Reset();
  }

  bool Valid() const override { return current_ < restarts_; }

  virtual void SeekToFirst() override final {
    SeekToFirstImpl();
    UpdateKey();
  }

  virtual void SeekToLast() override final {
    SeekToLastImpl();
    UpdateKey();
  }

  virtual void Seek(const Slice& target) override final {
    SeekImpl(target);
    UpdateKey();
  }

  virtual void SeekForPrev(const Slice& target) override final {
    SeekForPrevImpl(target);
    UpdateKey();
  }

  virtual void Next() override final {
    NextImpl();
    UpdateKey();
  }

  virtual bool NextAndGetResult(IterateResult* result) override final {
    // This does not need to call `UpdateKey()` as the parent class only has
    // access to the `UpdateKey()`-invoking functions.
    return InternalIteratorBase<TValue>::NextAndGetResult(result);
  }

  virtual void Prev() override final {
    PrevImpl();
    UpdateKey();
  }

  Status status() const override { return status_; }
  Slice key() const override {
    assert(Valid());
    return key_;
  }

#ifndef NDEBUG
  ~BlockIter() override {
    // Assert that the BlockIter is never deleted while Pinning is Enabled.
    assert(!pinned_iters_mgr_ ||
           (pinned_iters_mgr_ && !pinned_iters_mgr_->PinningEnabled()));
    status_.PermitUncheckedError();
  }
  void SetPinnedItersMgr(PinnedIteratorsManager* pinned_iters_mgr) override {
    pinned_iters_mgr_ = pinned_iters_mgr;
  }
  PinnedIteratorsManager* pinned_iters_mgr_ = nullptr;
#endif

  bool IsKeyPinned() const override {
    return block_contents_pinned_ && key_pinned_;
  }

  bool IsValuePinned() const override { return block_contents_pinned_; }

  size_t TEST_CurrentEntrySize() { return NextEntryOffset() - current_; }

  uint32_t ValueOffset() const {
    return static_cast<uint32_t>(value_.data() - data_);
  }

#ifdef VERSION_INDEX
  uint32_t CurrentKeyOffset() const {
    return raw_key_.GetKeyAddress() - data_;
  }

  int CompareKeyWithXmin(const Slice& other) {
    return memcmp(raw_key_.GetKeyAddress(), other.data(),
        other.size() - kNumInternalBytes + sizeof(SequenceNumber)); 
  }
#endif

  void SetCacheHandle(Cache::Handle* handle) { cache_handle_ = handle; }

  Cache::Handle* cache_handle() { return cache_handle_; }

 protected:
  std::unique_ptr<InternalKeyComparator> icmp_;
  const char* data_;       // underlying block contents
  uint32_t num_restarts_;  // Number of uint32_t entries in restart array

  // Index of restart block in which current_ or current_-1 falls
  uint32_t restart_index_;
  uint32_t restarts_;  // Offset of restart array (list of fixed32)
  // current_ is offset in data_ of current entry.  >= restarts_ if !Valid
  uint32_t current_;
  // Raw key from block.
  IterKey raw_key_;
  // Buffer for key data when global seqno assignment is enabled.
  IterKey key_buf_;
  Slice value_;
  Status status_;
  // Key to be exposed to users.
  Slice key_;
  bool key_pinned_;
  // Whether the block data is guaranteed to outlive this iterator, and
  // as long as the cleanup functions are transferred to another class,
  // e.g. PinnableSlice, the pointer to the bytes will still be valid.
  bool block_contents_pinned_;
  SequenceNumber global_seqno_;

  virtual void SeekToFirstImpl() = 0;
  virtual void SeekToLastImpl() = 0;
  virtual void SeekImpl(const Slice& target) = 0;
  virtual void SeekForPrevImpl(const Slice& target) = 0;
  virtual void NextImpl() = 0;

  virtual void PrevImpl() = 0;

  template <typename DecodeEntryFunc>
  inline bool ParseNextKey(bool* is_shared);

  void InitializeBase(const Comparator* raw_ucmp, const char* data,
                      uint32_t restarts, uint32_t num_restarts,
                      SequenceNumber global_seqno, bool block_contents_pinned) {
    assert(data_ == nullptr);  // Ensure it is called only once
    assert(num_restarts > 0);  // Ensure the param is valid

    icmp_ =
        std::make_unique<InternalKeyComparator>(raw_ucmp, false /* named */);
    data_ = data;
    restarts_ = restarts;
    num_restarts_ = num_restarts;
    current_ = restarts_;
    restart_index_ = num_restarts_;
    global_seqno_ = global_seqno;
    block_contents_pinned_ = block_contents_pinned;
    cache_handle_ = nullptr;
  }

  // Must be called every time a key is found that needs to be returned to user,
  // and may be called when no key is found (as a no-op). Updates `key_`,
  // `key_buf_`, and `key_pinned_` with info about the found key.
  void UpdateKey() {
    key_buf_.Clear();
    if (!Valid()) {
      return;
    }
    if (raw_key_.IsUserKey()) {
      assert(global_seqno_ == kDisableGlobalSequenceNumber);
      key_ = raw_key_.GetUserKey();
      key_pinned_ = raw_key_.IsKeyPinned();
    } else if (global_seqno_ == kDisableGlobalSequenceNumber) {
      key_ = raw_key_.GetInternalKey();
      key_pinned_ = raw_key_.IsKeyPinned();
    } else {
      key_buf_.SetInternalKey(raw_key_.GetUserKey(), global_seqno_,
                              ExtractValueType(raw_key_.GetInternalKey()));
      key_ = key_buf_.GetInternalKey();
      key_pinned_ = false;
    }
  }

  // Returns the result of `Comparator::Compare()`, where the appropriate
  // comparator is used for the block contents, the LHS argument is the current
  // key with global seqno applied, and the RHS argument is `other`.
  int CompareCurrentKey(const Slice& other) {
    if (raw_key_.IsUserKey()) {
      assert(global_seqno_ == kDisableGlobalSequenceNumber);
      return icmp_->user_comparator()->Compare(raw_key_.GetUserKey(), other);
    } else if (global_seqno_ == kDisableGlobalSequenceNumber) {
      return icmp_->Compare(raw_key_.GetInternalKey(), other);
    }
    return icmp_->Compare(raw_key_.GetInternalKey(), global_seqno_, other,
                          kDisableGlobalSequenceNumber);
  }

 private:
  // Store the cache handle, if the block is cached. We need this since the
  // only other place the handle is stored is as an argument to the Cleanable
  // function callback, which is hard to retrieve. When multiple value
  // PinnableSlices reference the block, they need the cache handle in order
  // to bump up the ref count
  Cache::Handle* cache_handle_;

 public:
  // Return the offset in data_ just past the end of the current entry.
  inline uint32_t NextEntryOffset() const {
    // NOTE: We don't support blocks bigger than 2GB
    return static_cast<uint32_t>((value_.data() + value_.size()) - data_);
  }

  uint32_t GetRestartPoint(uint32_t index) {
    assert(index < num_restarts_);
    return DecodeFixed32(data_ + restarts_ + index * sizeof(uint32_t));
  }

  void SeekToRestartPoint(uint32_t index) {
    raw_key_.Clear();
    restart_index_ = index;
    // current_ will be fixed by ParseNextKey();

    // ParseNextKey() starts at the end of value_, so set value_ accordingly
    uint32_t offset = GetRestartPoint(index);
    value_ = Slice(data_ + offset, 0);
  }

  void CorruptionError();

 protected:
  template <typename DecodeKeyFunc>
  inline bool BinarySeek(const Slice& target, uint32_t* index,
                         bool* is_index_key_result);

  void FindKeyAfterBinarySeek(const Slice& target, uint32_t index,
                              bool is_index_key_result);
};

class DataBlockIter final : public BlockIter<Slice> {
 public:
  DataBlockIter()
      : BlockIter(), read_amp_bitmap_(nullptr), last_bitmap_offset_(0) {}
  DataBlockIter(const Comparator* raw_ucmp, const char* data, uint32_t restarts,
                uint32_t num_restarts, SequenceNumber global_seqno,
                BlockReadAmpBitmap* read_amp_bitmap, bool block_contents_pinned,
                DataBlockHashIndex* data_block_hash_index)
      : DataBlockIter() {
    Initialize(raw_ucmp, data, restarts, num_restarts, global_seqno,
               read_amp_bitmap, block_contents_pinned, data_block_hash_index);
  }
  void Initialize(const Comparator* raw_ucmp, const char* data,
                  uint32_t restarts, uint32_t num_restarts,
                  SequenceNumber global_seqno,
                  BlockReadAmpBitmap* read_amp_bitmap,
                  bool block_contents_pinned,
                  DataBlockHashIndex* data_block_hash_index) {
    InitializeBase(raw_ucmp, data, restarts, num_restarts, global_seqno,
                   block_contents_pinned);
    raw_key_.SetIsUserKey(false);
    read_amp_bitmap_ = read_amp_bitmap;
    last_bitmap_offset_ = current_ + 1;
    data_block_hash_index_ = data_block_hash_index;
  }

  Slice value() const override {
    assert(Valid());
    if (read_amp_bitmap_ && current_ < restarts_ &&
        current_ != last_bitmap_offset_) {
      read_amp_bitmap_->Mark(current_ /* current entry offset */,
                             NextEntryOffset() - 1);
      last_bitmap_offset_ = current_;
    }
    return value_;
  }

  inline bool SeekForGet(const Slice& target) {
    if (!data_block_hash_index_) {
      SeekImpl(target);
      UpdateKey();
      return true;
    }
    bool res = SeekForGetImpl(target);
    UpdateKey();
    return res;
  }

  void Invalidate(const Status& s) override {
    BlockIter::Invalidate(s);
    // Clear prev entries cache.
    prev_entries_keys_buff_.clear();
    prev_entries_.clear();
    prev_entries_idx_ = -1;
  }

 protected:
  friend Block;
  inline bool ParseNextDataKey(bool* is_shared);
  void SeekToFirstImpl() override;
  void SeekToLastImpl() override;
  void SeekImpl(const Slice& target) override;
  void SeekForPrevImpl(const Slice& target) override;
  void NextImpl() override;
  void PrevImpl() override;

 private:
  // read-amp bitmap
  BlockReadAmpBitmap* read_amp_bitmap_;
  // last `current_` value we report to read-amp bitmp
  mutable uint32_t last_bitmap_offset_;
  struct CachedPrevEntry {
    explicit CachedPrevEntry(uint32_t _offset, const char* _key_ptr,
                             size_t _key_offset, size_t _key_size, Slice _value)
        : offset(_offset),
          key_ptr(_key_ptr),
          key_offset(_key_offset),
          key_size(_key_size),
          value(_value) {}

    // offset of entry in block
    uint32_t offset;
    // Pointer to key data in block (nullptr if key is delta-encoded)
    const char* key_ptr;
    // offset of key in prev_entries_keys_buff_ (0 if key_ptr is not nullptr)
    size_t key_offset;
    // size of key
    size_t key_size;
    // value slice pointing to data in block
    Slice value;
  };
  std::string prev_entries_keys_buff_;
  std::vector<CachedPrevEntry> prev_entries_;
  int32_t prev_entries_idx_ = -1;

  DataBlockHashIndex* data_block_hash_index_;

  bool SeekForGetImpl(const Slice& target);
};

// Iterator over MetaBlocks.  MetaBlocks are similar to Data Blocks and
// are used to store Properties associated with table.
// Meta blocks always store user keys (no sequence number) and always
// use the BytewiseComparator.  Additionally, MetaBlock accesses are
// not recorded in the Statistics or for Read-Amplification.
class MetaBlockIter final : public BlockIter<Slice> {
 public:
  MetaBlockIter() : BlockIter() { raw_key_.SetIsUserKey(true); }
  void Initialize(const char* data, uint32_t restarts, uint32_t num_restarts,
                  bool block_contents_pinned) {
    // Initializes the iterator with a BytewiseComparator and
    // the raw key being a user key.
    InitializeBase(BytewiseComparator(), data, restarts, num_restarts,
                   kDisableGlobalSequenceNumber, block_contents_pinned);
    raw_key_.SetIsUserKey(true);
  }

  Slice value() const override {
    assert(Valid());
    return value_;
  }

 protected:
  void SeekToFirstImpl() override;
  void SeekToLastImpl() override;
  void SeekImpl(const Slice& target) override;
  void SeekForPrevImpl(const Slice& target) override;
  void NextImpl() override;
  void PrevImpl() override;
};

class IndexBlockIter final : public BlockIter<IndexValue> {
 public:
  IndexBlockIter() : BlockIter(), prefix_index_(nullptr) {}

  // key_includes_seq, default true, means that the keys are in internal key
  // format.
  // value_is_full, default true, means that no delta encoding is
  // applied to values.
  void Initialize(const Comparator* raw_ucmp, const char* data,
                  uint32_t restarts, uint32_t num_restarts,
                  SequenceNumber global_seqno, BlockPrefixIndex* prefix_index,
                  bool have_first_key, bool key_includes_seq,
                  bool value_is_full, bool block_contents_pinned) {
    InitializeBase(raw_ucmp, data, restarts, num_restarts,
                   kDisableGlobalSequenceNumber, block_contents_pinned);
    raw_key_.SetIsUserKey(!key_includes_seq);
    prefix_index_ = prefix_index;
    value_delta_encoded_ = !value_is_full;
    have_first_key_ = have_first_key;
    if (have_first_key_ && global_seqno != kDisableGlobalSequenceNumber) {
      global_seqno_state_.reset(new GlobalSeqnoState(global_seqno));
    } else {
      global_seqno_state_.reset();
    }
  }

  Slice user_key() const override {
    assert(Valid());
    return raw_key_.GetUserKey();
  }

  IndexValue value() const override {
    assert(Valid());
    if (value_delta_encoded_ || global_seqno_state_ != nullptr) {
      return decoded_value_;
    } else {
      IndexValue entry;
      Slice v = value_;
      Status decode_s __attribute__((__unused__)) =
          entry.DecodeFrom(&v, have_first_key_, nullptr);
      assert(decode_s.ok());
      return entry;
    }
  }

  bool IsValuePinned() const override {
    return global_seqno_state_ != nullptr ? false : BlockIter::IsValuePinned();
  }

 protected:
  // IndexBlockIter follows a different contract for prefix iterator
  // from data iterators.
  // If prefix of the seek key `target` exists in the file, it must
  // return the same result as total order seek.
  // If the prefix of `target` doesn't exist in the file, it can either
  // return the result of total order seek, or set both of Valid() = false
  // and status() = NotFound().
  void SeekImpl(const Slice& target) override;

  void SeekForPrevImpl(const Slice&) override {
    assert(false);
    current_ = restarts_;
    restart_index_ = num_restarts_;
    status_ = Status::InvalidArgument(
        "RocksDB internal error: should never call SeekForPrev() on index "
        "blocks");
    raw_key_.Clear();
    value_.clear();
  }

  void PrevImpl() override;

  void NextImpl() override;

  void SeekToFirstImpl() override;

  void SeekToLastImpl() override;

 private:
  bool value_delta_encoded_;
  bool have_first_key_;  // value includes first_internal_key
  BlockPrefixIndex* prefix_index_;
  // Whether the value is delta encoded. In that case the value is assumed to be
  // BlockHandle. The first value in each restart interval is the full encoded
  // BlockHandle; the restart of encoded size part of the BlockHandle. The
  // offset of delta encoded BlockHandles is computed by adding the size of
  // previous delta encoded values in the same restart interval to the offset of
  // the first value in that restart interval.
  IndexValue decoded_value_;

  // When sequence number overwriting is enabled, this struct contains the seqno
  // to overwrite with, and current first_internal_key with overwritten seqno.
  // This is rarely used, so we put it behind a pointer and only allocate when
  // needed.
  struct GlobalSeqnoState {
    // First internal key according to current index entry, but with sequence
    // number overwritten to global_seqno.
    IterKey first_internal_key;
    SequenceNumber global_seqno;

    explicit GlobalSeqnoState(SequenceNumber seqno) : global_seqno(seqno) {}
  };

  std::unique_ptr<GlobalSeqnoState> global_seqno_state_;

  // Set *prefix_may_exist to false if no key possibly share the same prefix
  // as `target`. If not set, the result position should be the same as total
  // order Seek.
  bool PrefixSeek(const Slice& target, uint32_t* index, bool* prefix_may_exist);
  // Set *prefix_may_exist to false if no key can possibly share the same
  // prefix as `target`. If not set, the result position should be the same
  // as total order seek.
  bool BinaryBlockIndexSeek(const Slice& target, uint32_t* block_ids,
                            uint32_t left, uint32_t right, uint32_t* index,
                            bool* prefix_may_exist);
  inline int CompareBlockKey(uint32_t block_index, const Slice& target);

  inline bool ParseNextIndexKey();

  // When value_delta_encoded_ is enabled it decodes the value which is assumed
  // to be BlockHandle and put it to decoded_value_
  inline void DecodeCurrentValue(bool is_shared);
};


#ifdef SMARTSSD
class NDPResultsIter final : public InternalIteratorBase<Slice> {
 public:
  /* In sysbench key_size_ is 28 */
  NDPResultsIter()
    : InternalIteratorBase() {
    data_ = nullptr; 
    next_data_ = nullptr;
    test_file_ = nullptr;
    es_ndpo_ = nullptr;
  }

  ~NDPResultsIter() {
    if (data_)
      free(data_);
    if (next_data_)
      free(next_data_);
    if (test_file_)
      std::fclose(test_file_);
  }

  void Initialize(void* es_ndpo, int cf_id, uint64_t key_size, 
      const Comparator* raw_ucmp = BytewiseComparator()) {
    data_ = nullptr;
    next_data_ = nullptr;
    test_file_ = nullptr;
    es_ndpo_ = es_ndpo;
    cf_id_  = cf_id;
    key_size_ = key_size;
    value_prepared_ = false;
    icmp_ = std::make_unique<InternalKeyComparator>(raw_ucmp, false /*named*/);
    num_rec_ = 0;
    cur_num_rec_ = -1;
    next_num_rec_ = 0;
    raw_key_.Clear();
    cur_block_num_ = -1; 
    next_block_num_ = 0;
    rested_data_ = true;
  }

  // An iterator is either positioned at a key/value pair, or
  // not valid.  This method returns true iff the iterator is valid.
  // Always returns false if !status().ok().
  bool Valid() const override {
    return rested_data_;
  }

  void MoveNext() {
    if (cur_block_num_ != next_block_num_) {
      // Read block
      // Set Next Block
      SetNextBlock();
      cur_block_num_ = next_block_num_;
    } 
    cur_num_rec_ = next_num_rec_;
  }

  inline uint32_t start_offset() {
    return sizeof(num_rec_) + sizeof(rec_size_);
  }

  uint32_t GetCurrentPoint() {
    return start_offset() + rec_size_ * cur_num_rec_;
  }

  void SetCurrentRecord(uint32_t num_rec) {
    cur_num_rec_ = num_rec;
  }

  uint32_t GetRecordOffset(uint32_t rec_idx) {
    SetCurrentRecord(rec_idx);
    return GetCurrentPoint();
  }

  uint32_t GetRecordOffsetWoChgIdx(uint32_t rec_idx) {
    return start_offset() + rec_size_ * rec_idx;
  }

  void UpdateKey() {
    if (!Valid()) {
      return;
    }
    if (raw_key_.IsUserKey()) {
      key_ = raw_key_.GetUserKey();
      key_pinned_ = raw_key_.IsKeyPinned();
    } else {
      key_ = raw_key_.GetInternalKey();
      key_pinned_ = raw_key_.IsKeyPinned();
    }
  }

  bool ParseNextDataKey(const char* p) {
    if (!Valid()) {
      value_prepared_ = false;
      return false;
    }

    //Decode next entry
    raw_key_.SetKey(Slice(p, key_size_), false /* copy */);
    value_ = Slice(p + key_size_, val_size_);

    next_num_rec_ = cur_num_rec_ + 1;

    if (next_num_rec_ == num_rec_) {
      next_block_num_ = cur_block_num_ + 1;
      // Ready Next Block
      ReadyNextBlock();
      next_num_rec_ = 0;
    }

    value_prepared_ = true;

    return true;
  }

  bool ParseNextDataKey(const char* p, uint16_t my_key_size) {
    if (!Valid()) {
      value_prepared_ = false;
      return false;
    }

    //Decode next entry
    raw_key_.SetKey(Slice(p, my_key_size), false /* copy */);
    value_ = Slice(p + key_size_, val_size_);

    next_num_rec_ = cur_num_rec_ + 1;

    // Ready Next Block
    if (next_num_rec_ == num_rec_) {
      next_block_num_ = cur_block_num_ + 1;

      ReadyNextBlock();
      next_num_rec_ = 0;
    }

    value_prepared_ = true;

    return true;
  }


  // Position at the first key in the source.  The iterator is Valid()
  // after this call iff the source is not empty.
  void SeekToFirst() override {
    raw_key_.Clear();

    if (cur_block_num_ != 0) {
      next_block_num_ = cur_block_num_ = 0;
      ReadyNextBlock();
      SetNextBlock();
    }
    uint32_t offset = GetRecordOffset(0);

    value_ = Slice(data_ + offset, rec_size_);
    if (!ParseNextDataKey(data_ + offset)) {
      return;
    }

    UpdateKey();
  }

  // Position at the last key in the source.  The iterator is
  // Valid() after this call iff the source is not empty.
  void SeekToLast() override {
    SeekToFirst();
  }

  // Position at the first key in the source that at or past target
  // The iterator is Valid() after this call iff the source contains
  // an entry that comes at or past target.
  // All Seek*() methods clear any error status() that the iterator had prior to
  // the call; after the seek, status() indicates only the error (if any) that
  // happened during the seek, not any past errors.
  // 'target' contains user timestamp if timestamp is enabled.
  void Seek(const Slice& target) override {
    rested_data_ = true;
    block_read_ret_ = 0;
    cur_block_num_ = 0;
    next_block_num_ = 0;

    ReadyNextBlock();
    SetNextBlock();
    SeekToFirst();
    return;

    /*
    uint16_t target_key_size = target.size();

    SeekToFirst();

    uint32_t offset = GetRecordOffset(0);
    
    raw_key_.Clear();
    while (ParseNextDataKey(data_ + offset, target_key_size)) {
      value_ = Slice(data_ + offset, rec_size_);

      if (raw_key_.IsUserKey()) {
        if (!icmp_->Compare(raw_key_.GetUserKey(), target)) {
          UpdateKey();
          // valid_ = true;
          return;
        } else if (icmp_->Compare(raw_key_.GetUserKey(), target) < 0) {
          SeekToFirst();
          return;
        } 
      } else {
        if (!icmp_->Compare(raw_key_.GetInternalKey(), target)) {
          UpdateKey();
          // valid_ = true;
          return;
        } else if (icmp_->Compare(raw_key_.GetInternalKey(), target) < 0) {
          SeekToFirst();
          return;
        } 
      }
      MoveNext();
      offset = GetCurrentPoint();
      raw_key_.Clear();
    }
  */
  }

  // Position at the first key in the source that at or before target
  // The iterator is Valid() after this call iff the source contains
  // an entry that comes at or before target.
  void SeekForPrev(const Slice& target) override {
    Seek(target);
  }

  // Moves to the next entry in the source.  After this call, Valid() is
  // true iff the iterator was not positioned at the last entry in the source.
  // REQUIRES: Valid()
  void Next() override {
    MoveNext();
    ParseNextDataKey(data_ + GetCurrentPoint());
    UpdateKey();
  }

  // Moves to the next entry in the source, and return result. Iterator
  // implementation should override this method to help methods inline better,
  // or when UpperBoundCheckResult() is non-trivial.
  // REQUIRES: Valid()

  bool NextAndGetResult(IterateResult* result) override {
    // This does not need to call `UpdateKey()` as the parent class only has
    // access to the `UpdateKey()`-invoking functions.
    Next();
    bool is_valid = Valid();
    if (is_valid) {
      result->key = key();
      // Default may_be_out_of_upper_bound to true to avoid unnecessary virtual
      // call. If an implementation has non-trivial UpperBoundCheckResult(),
      // it should also override NextAndGetResult().
      result->bound_check_result = IterBoundCheck::kInbound;
      result->value_prepared = true;
    } else {
      result->bound_check_result = IterBoundCheck::kOutOfBound;
      result->value_prepared = false;
    }
    return is_valid;
  }

  // Moves to the previous entry in the source.  After this call, Valid() is
  // true iff the iterator was not positioned at the first entry in source.
  // REQUIRES: Valid()
  void Prev() override {
    Next();
  }

  // Return the key for the current entry.  The underlying storage for
  // the returned slice is valid only until the next modification of
  // the iterator.
  // REQUIRES: Valid()
  Slice key() const {
    assert(Valid());
    return key_;
  }

  // Return user key for the current entry.
  // REQUIRES: Valid()
  Slice user_key() const {
    assert(Valid());
    //assert(value_prepared_);
    return InternalIteratorBase<Slice>::user_key();
  }

  // Return the value for the current entry.  The underlying storage for
  // the returned slice is valid only until the next modification of
  // the iterator.
  // REQUIRES: Valid()
  // REQUIRES: PrepareValue() has been called if needed (see PrepareValue()).
  Slice value() const override {
    assert(Valid());
    assert(value_prepared_);
    return value_;
  }

  // If an error has occurred, return it.  Else return an ok status.
  // If non-blocking IO is requested and this operation cannot be
  // satisfied without doing some IO, then this returns Status::Incomplete().
  Status status() const override { return status_; }

  // For some types of iterators, sometimes Seek()/Next()/SeekForPrev()/etc may
  // load key but not value (to avoid the IO cost of reading the value from disk
  // if it won't be not needed). This method loads the value in such situation.
  //
  // Needs to be called before value() at least once after each iterator
  // movement (except if IterateResult::value_prepared = true), for iterators
  // created with allow_unprepared_value = true.
  //
  // Returns false if an error occurred; in this case Valid() is also changed
  // to false, and status() is changed to non-ok.
  // REQUIRES: Valid()
  bool PrepareValue() {
    if (!Valid()) {
      return false;
    }

    value_ = Slice(data_ + GetCurrentPoint() + key_size_, val_size_);
    return true;
  }

  // Set next block
  void SetNextBlock() {
    char *for_toggle;
    if (block_read_ret_ >= 0) {
      for_toggle = data_;
      data_ = next_data_;
      next_data_ = for_toggle;
    } else {
      rested_data_ = false;
      free(data_);
      free(next_data_);
      data_ = nullptr;
      next_data_ = nullptr;
    }
  }

  // Ready next block when next record is in another block
  bool ReadyNextBlock() { 
    int ret = -1;

    if (data_ == nullptr) {
      data_ =  static_cast<char*>(malloc(4UL << 20));
      next_data_ =  static_cast<char*>(malloc(4UL << 20));
    }
#if 1
    if (block_read_ret_ < 0) {
      rested_data_ = false;
      return false;
    }

    while ((ret = Get_Packets(es_ndpo_, cf_id_, next_data_, next_block_num_)) 
        == -2);

    block_read_ret_ = ret;

    if (ret < 0) {
#ifdef NDP_DEBUG
      fprintf(stderr, "SMARTSSD read: %d (%dth block) %d fail \n",
          ret, next_block_num_, cf_id_);
#endif //NDP_DEBUG
      return false;
    } else {
      num_rec_ = *((int32_t*) next_data_);
      rec_size_ = *((int16_t*) (next_data_ + sizeof(num_rec_)));
      val_size_ = rec_size_ - key_size_;
#ifdef NDP_DEBUG
      fprintf(stderr, "SMARTSSD read out %dth block from %d\n",
          next_block_num_, cf_id_);
      fprintf(stderr, "SMARTSSD read: %d (%dth block) from %d (cur: %u, rec: %d)\n",
          ret, next_block_num_, cf_id_, next_num_rec_, num_rec_);
#endif //NDP_DEBUG
      return true;
    }
/******** Previous code in PrepareValue() **********/
/***********
    uint32_t offset = GetRecordOffsetWoChgIdx(cur_num_rec_);
    char *s3d_buff = nullptr;

    if (data_) {
      s3d_buff = data_;
    } else {
      s3d_buff = static_cast<char *>(malloc(4UL << 20));
    }
    status_ = Status::OK();
    if (next_) {
//#ifdef WITH_NDP_LIB
      while ((ret = Get_Packets(es_ndpo_, cf_id_, s3d_buff, block_num_)) == -2);
//#endif //WITH_NDP_LIB
      fprintf(stderr, "SMARTSSD read: %d (%dth block) from %d (cur: %u, rec: %u)\n",
          ret, block_num_, cf_id_, cur_num_rec_, num_rec_);
    }

    if (ret < 0) {
      next_ = false;
      return true;
    } else {
      data_ = s3d_buff;
      cur_num_rec_ = 0;
      num_rec_ = *((int32_t*) data_);
      rec_size_ = *((int16_t*) (data_ + sizeof(num_rec_)));
      val_size_ = rec_size_ - key_size_;
      value_prepared_ = true;
      block_num_++;
      return true;
    }
*********/
#else
    char filename[64];
    std::sprintf(filename, "/tmp/ndp.%d", cf_id_);

    if (!test_file_) {
      test_file_ = std::fopen(filename, "r");
      cur_block_num_ = 0;
      next_block_num_ = 0;
      // std::fseek(test_file_, 0, SEEK_END);
    }

    if (block_read_ret_ < 0) {
      return false;
    }

    std::fseek(test_file_, 0, SEEK_END);
    size_t total_len = std::ftell(test_file_);
    size_t remained, read;

    if (!(ret = std::fseek(test_file_, next_block_num_ * (4UL << 20), 
        SEEK_SET))) {
      size_t cur_len = std::ftell(test_file_);

      if (total_len == cur_len) {
        /*
        fprintf(stderr, "SMARTSSD read: %lu (%dth block) %d file end\n",
            ret, next_block_num_, cf_id_);
            */
        block_read_ret_ = -1;
        return false;
      }

      /*
      fprintf(stderr, "SMARTSSD read out %dth block from %d\n",
          next_block_num_, cf_id_);
          */

      next_data_ = static_cast<char *>(malloc(4UL << 20));

      read = 0;
      remained = (total_len - cur_len >= (4UL << 20)) ? (4UL << 20) :
          total_len - cur_len;
      size_t write_size = remained;
      do {
        if ((read = std::fread(next_data_ + write_size - remained, 1,
            remained, test_file_)) == 0) {
          ret = remained > 0;
          break;
        }
      } while ((remained -= read) > 0);
    }

    if (ret) {
      /*
      fprintf(stderr, "SMARTSSD read: %lu (%dth block) %d fail \n",
          ret, next_block_num_, cf_id_);
          */
      block_read_ret_ = -1;
      return false;
    } else {
      block_read_ret_ = 0;
      num_rec_ = *((int32_t*) next_data_);
      rec_size_ = *((int16_t*) (next_data_ + sizeof(num_rec_)));
      val_size_ = rec_size_ - key_size_;
      /*
      fprintf(stderr, "SMARTSSD read: %lu (%dth block) from %d (cur: %u, rec: %u)\n",
          ret, next_block_num_, cf_id_, cur_num_rec_, num_rec_);
          */
      return true;
    }
#endif
  }


#if 0
bool FetchNextNDPBlock() {
    int ret = -1;
#if 0
    uint32_t offset = GetRecordOffsetWoChgIdx(cur_num_rec_);
    char *s3d_buff = nullptr;

    if (data_) {
      s3d_buff = data_;
    } else {
      s3d_buff = static_cast<char *>(malloc(4UL << 20));
    }

    status_ = Status::OK();

    if (next_) {
//#ifdef WITH_NDP_LIB
      while ((ret = Get_Packets(es_ndpo_, cf_id_, s3d_buff, block_num_)) == -2);
//#endif //WITH_NDP_LIB
      fprintf(stderr, "SMARTSSD read: %d (%dth block) from %d (cur: %u, rec: %u)\n", 
          ret, block_num_, cf_id_, cur_num_rec_, num_rec_);
    }

    if (ret < 0) {
      next_ = false;
      return true;
    } else {
      data_ = s3d_buff;
      cur_num_rec_ = 0;
      num_rec_ = *((int32_t*) data_);
      rec_size_ = *((int16_t*) (data_ + sizeof(num_rec_)));
      val_size_ = rec_size_ - key_size_;
      value_prepared_ = true;
      block_num_++;
      return true;
    }
#else
    // TODO: Using Get_Packets

    FILE* f;
    size_t remained, read;
    if (next_) {
      char filename[64];
      std::sprintf(filename, "/tmp/ndp.%d", cf_id_);
      f = std::fopen(filename, "r");
      fprintf(stderr, "SMARTSSD open the file (%s) of %d\n", filename, cf_id_);
      if (!(ret = std::fseek(f, block_num_ * (4UL << 20), SEEK_SET))) {
        fprintf(stderr, "SMARTSSD read out %dth block from %d\n", block_num_, cf_id_);
        if (data_ == nullptr) {
          data_ = static_cast<char *>(malloc(4UL << 20));
        }
        read = 0;
        remained = 4UL << 20;
        do {
          if ((read = std::fread(data_, 1, 4UL << 20, f)) == 0) {
            ret = remained > 0;
            break;
          }
        } while ((remained -= read) > 0);
      }
      std::fclose(f);
    }

    if (ret) {
      // error
      valid_ = false;
      next_ = false;
      value_prepared_ = false;
      return false;
    } else {
      //data_ = s3d_buff;
      valid_ = true;
      cur_num_rec_ = -1;
      num_rec_ = *((int32_t*) data_);
      rec_size_ = *((int16_t*) (data_ + sizeof(num_rec_)));
      val_size_ = rec_size_ - key_size_;
      value_prepared_ = true;
      block_num_++;
      return true;
    }
#endif 
  }
#endif

  // Keys return from this iterator can be smaller than iterate_lower_bound.
  bool MayBeOutOfLowerBound() { 
    return rested_data_;
  }

  // If the iterator has checked the key against iterate_upper_bound, returns
  // the result here. The function can be used by user of the iterator to skip
  // their own checks. If Valid() = true, IterBoundCheck::kUnknown is always
  // a valid value. If Valid() = false, IterBoundCheck::kOutOfBound indicates
  // that the iterator is filtered out by upper bound checks.
  IterBoundCheck UpperBoundCheckResult() {
    if (rested_data_) {
      return IterBoundCheck::kInbound;
    } else {
      return IterBoundCheck::kOutOfBound;
    }
  }

  // Pass the PinnedIteratorsManager to the Iterator, most Iterators don't
  // communicate with PinnedIteratorsManager so default implementation is no-op
  // but for Iterators that need to communicate with PinnedIteratorsManager
  // they will implement this function and use the passed pointer to communicate
  // with PinnedIteratorsManager.
  void SetPinnedItersMgr(PinnedIteratorsManager* /*pinned_iters_mgr*/) {
  }

  // If true, this means that the Slice returned by key() is valid as long as
  // PinnedIteratorsManager::ReleasePinnedData is not called and the
  // Iterator is not deleted.
  //
  // IsKeyPinned() is guaranteed to always return true if
  //  - Iterator is created with ReadOptions::pin_data = true
  //  - DB tables were created with BlockBasedTableOptions::use_delta_encoding
  //    set to false.
  bool IsKeyPinned() const { return key_pinned_; }

  // If true, this means that the Slice returned by value() is valid as long as
  // PinnedIteratorsManager::ReleasePinnedData is not called and the
  // Iterator is not deleted.
  // REQUIRES: Same as for value().
  bool IsValuePinned() const { return false; }

  Status GetProperty(std::string /*prop_name*/, std::string* /*prop*/) {
    return Status::NotSupported("");
  }

  // When iterator moves from one file to another file at same level, new file's
  // readahead state (details of last block read) is updated with previous
  // file's readahead state. This way internal readahead_size of Prefetch Buffer
  // doesn't start from scratch and can fall back to 8KB with no prefetch if
  // reads are not sequential.
  //
  // Default implementation is no-op and its implemented by iterators.
  void GetReadaheadState(ReadaheadFileInfo* /*readahead_file_info*/) {}

  // Default implementation is no-op and its implemented by iterators.
  void SetReadaheadState(ReadaheadFileInfo* /*readahead_file_info*/) {}

 protected:
  void SeekForPrevImpl(const Slice& target, const Comparator* cmp) {
    Seek(target);
    if (!Valid()) {
      SeekToLast();
    }
    while (Valid() && cmp->Compare(target, key()) < 0) {
      Prev();
    }
  }
 private:
  char* data_; /* ndp buffer data */
  char* next_data_; /* ndp buffer data */
  void* es_ndpo_; /*cached NDP Object */

  int16_t block_num_; /* used for calculation of 4 MiB offset in NDP Buffers */
  int16_t rec_size_; /* size of record */
  int16_t key_size_; /* size of key */
  int16_t val_size_; /* size of value */

  int cf_id_; /* associated column family id for this iterator */
  int table_index_; /* associated table_index (NDP internal use) */

  int32_t num_rec_; /* number of total record in this buffer */
  int32_t cur_num_rec_; /* current index of record */
  int32_t next_num_rec_; /* next index of record */

  int16_t cur_block_num_;
  int16_t next_block_num_;
  int block_read_ret_;

  bool rested_data_;

  IterKey raw_key_; /* raw key */

  Slice key_;
  Slice value_;
  Status status_;

  bool key_pinned_;
  bool value_prepared_;
  std::unique_ptr<InternalKeyComparator> icmp_;
  FILE* test_file_;
};
#endif /* SMARTSSD */

}  // namespace ROCKSDB_NAMESPACE
