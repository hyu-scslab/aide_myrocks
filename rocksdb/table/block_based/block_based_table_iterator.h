//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
#pragma once
#include "table/block_based/block.h"
#include "table/block_based/block_based_table_reader.h"
#include "table/block_based/block_based_table_reader_impl.h"
#include "table/block_based/block_prefetcher.h"
#include "table/block_based/reader_common.h"

namespace ROCKSDB_NAMESPACE {

// Iterates over the contents of BlockBasedTable.
class BlockBasedTableIterator : public InternalIteratorBase<Slice> {
  // compaction_readahead_size: its value will only be used if for_compaction =
  // true
  // @param read_options Must outlive this iterator.
 public:
  BlockBasedTableIterator(
      const BlockBasedTable* table, const ReadOptions& read_options,
      const InternalKeyComparator& icomp,
      std::unique_ptr<InternalIteratorBase<IndexValue>>&& index_iter,
      bool check_filter, bool need_upper_bound_check,
      const SliceTransform* prefix_extractor, TableReaderCaller caller,
      size_t compaction_readahead_size = 0, bool allow_unprepared_value = false)
      : index_iter_(std::move(index_iter)),
        table_(table),
        read_options_(read_options),
        icomp_(icomp),
        user_comparator_(icomp.user_comparator()),
        pinned_iters_mgr_(nullptr),
        prefix_extractor_(prefix_extractor),
        lookup_context_(caller),
        block_prefetcher_(
            compaction_readahead_size,
            table_->get_rep()->table_options.initial_auto_readahead_size),
        allow_unprepared_value_(allow_unprepared_value),
        block_iter_points_to_real_block_(false),
        check_filter_(check_filter),
        need_upper_bound_check_(need_upper_bound_check),
        async_read_in_progress_(false) {
  }

  ~BlockBasedTableIterator() {}

  void Seek(const Slice& target) override;
  void SeekForPrev(const Slice& target) override;
  void SeekToFirst() override;
  void SeekToLast() override;
  void Next() final override;
  bool NextAndGetResult(IterateResult* result) override;
  void Prev() override;
  bool Valid() const override {
    return !is_out_of_bound_ &&
           (is_at_first_key_from_index_ ||
            (block_iter_points_to_real_block_ && block_iter_.Valid()));
  }
  Slice key() const override {
    assert(Valid());
    if (is_at_first_key_from_index_) {
      return index_iter_->value().first_internal_key;
    } else {
      return block_iter_.key();
    }
  }
  Slice user_key() const override {
    assert(Valid());
    if (is_at_first_key_from_index_) {
      return ExtractUserKey(index_iter_->value().first_internal_key);
    } else {
      return block_iter_.user_key();
    }
  }
  bool PrepareValue() override {
    assert(Valid());
    if (!is_at_first_key_from_index_) {
      return true;
    }

    return const_cast<BlockBasedTableIterator*>(this)
        ->MaterializeCurrentBlock();
  }
  Slice value() const override {
    // PrepareValue() must have been called.
    assert(!is_at_first_key_from_index_);
    assert(Valid());

    return block_iter_.value();
  }
  Status status() const override {
    // Prefix index set status to NotFound when the prefix does not exist
    if (!index_iter_->status().ok() && !index_iter_->status().IsNotFound()) {
      return index_iter_->status();
    } else if (block_iter_points_to_real_block_) {
      return block_iter_.status();
    } else if (async_read_in_progress_) {
      return Status::TryAgain();
    } else {
      return Status::OK();
    }
  }

  inline IterBoundCheck UpperBoundCheckResult() override {
    if (is_out_of_bound_) {
      return IterBoundCheck::kOutOfBound;
    } else if (block_upper_bound_check_ ==
               BlockUpperBound::kUpperBoundBeyondCurBlock) {
      assert(!is_out_of_bound_);
      return IterBoundCheck::kInbound;
    } else {
      return IterBoundCheck::kUnknown;
    }
  }

  void SetPinnedItersMgr(PinnedIteratorsManager* pinned_iters_mgr) override {
    pinned_iters_mgr_ = pinned_iters_mgr;
  }
  bool IsKeyPinned() const override {
    // Our key comes either from block_iter_'s current key
    // or index_iter_'s current *value*.
    return pinned_iters_mgr_ && pinned_iters_mgr_->PinningEnabled() &&
           ((is_at_first_key_from_index_ && index_iter_->IsValuePinned()) ||
            (block_iter_points_to_real_block_ && block_iter_.IsKeyPinned()));
  }
  bool IsValuePinned() const override {
    assert(!is_at_first_key_from_index_);
    assert(Valid());

    // BlockIter::IsValuePinned() is always true. No need to check
    return pinned_iters_mgr_ && pinned_iters_mgr_->PinningEnabled() &&
           block_iter_points_to_real_block_;
  }

  void ResetDataIter() {
    if (block_iter_points_to_real_block_) {
      if (pinned_iters_mgr_ != nullptr && pinned_iters_mgr_->PinningEnabled()) {
        block_iter_.DelegateCleanupsTo(pinned_iters_mgr_);
      }
      block_iter_.Invalidate(Status::OK());
      block_iter_points_to_real_block_ = false;
    }
    block_upper_bound_check_ = BlockUpperBound::kUnknown;
  }

  void SavePrevIndexValue() {
    if (block_iter_points_to_real_block_) {
      // Reseek. If they end up with the same data block, we shouldn't re-fetch
      // the same data block.
      prev_block_offset_ = index_iter_->value().handle.offset();
    }
  }

  void GetReadaheadState(ReadaheadFileInfo* readahead_file_info) override {
    if (block_prefetcher_.prefetch_buffer() != nullptr &&
        read_options_.adaptive_readahead) {
      block_prefetcher_.prefetch_buffer()->GetReadaheadState(
          &(readahead_file_info->data_block_readahead_info));
      if (index_iter_) {
        index_iter_->GetReadaheadState(readahead_file_info);
      }
    }
  }

  void SetReadaheadState(ReadaheadFileInfo* readahead_file_info) override {
    if (read_options_.adaptive_readahead) {
      block_prefetcher_.SetReadaheadState(
          &(readahead_file_info->data_block_readahead_info));
      if (index_iter_) {
        index_iter_->SetReadaheadState(readahead_file_info);
      }
    }
  }

#ifdef VERSION_INDEX
  uint32_t Offset() {
    return index_iter_->value().handle.offset()
      + block_iter_.CurrentKeyOffset();
  }

  int CompareKeyWithXmin(const Slice& other) {
    return block_iter_.CompareKeyWithXmin(other);
  }
#endif

  std::unique_ptr<InternalIteratorBase<IndexValue>> index_iter_;

 private:
  enum class IterDirection {
    kForward,
    kBackward,
  };
  // This enum indicates whether the upper bound falls into current block
  // or beyond.
  //   +-------------+
  //   |  cur block  |       <-- (1)
  //   +-------------+
  //                         <-- (2)
  //  --- <boundary key> ---
  //                         <-- (3)
  //   +-------------+
  //   |  next block |       <-- (4)
  //        ......
  //
  // When the block is smaller than <boundary key>, kUpperBoundInCurBlock
  // is the value to use. The examples are (1) or (2) in the graph. It means
  // all keys in the next block or beyond will be out of bound. Keys within
  // the current block may or may not be out of bound.
  // When the block is larger or equal to <boundary key>,
  // kUpperBoundBeyondCurBlock is to be used. The examples are (3) and (4)
  // in the graph. It means that all keys in the current block is within the
  // upper bound and keys in the next block may or may not be within the uppder
  // bound.
  // If the boundary key hasn't been checked against the upper bound,
  // kUnknown can be used.
  enum class BlockUpperBound {
    kUpperBoundInCurBlock,
    kUpperBoundBeyondCurBlock,
    kUnknown,
  };

  const BlockBasedTable* table_;
  const ReadOptions& read_options_;
  const InternalKeyComparator& icomp_;
  UserComparatorWrapper user_comparator_;
  PinnedIteratorsManager* pinned_iters_mgr_;
  DataBlockIter block_iter_;
  const SliceTransform* prefix_extractor_;
  uint64_t prev_block_offset_ = std::numeric_limits<uint64_t>::max();
  BlockCacheLookupContext lookup_context_;

  BlockPrefetcher block_prefetcher_;

  const bool allow_unprepared_value_;
  // True if block_iter_ is initialized and points to the same block
  // as index iterator.
  bool block_iter_points_to_real_block_;
  // See InternalIteratorBase::IsOutOfBound().
  bool is_out_of_bound_ = false;
  // How current data block's boundary key with the next block is compared with
  // iterate upper bound.
  BlockUpperBound block_upper_bound_check_ = BlockUpperBound::kUnknown;
  // True if we're standing at the first key of a block, and we haven't loaded
  // that block yet. A call to PrepareValue() will trigger loading the block.
  bool is_at_first_key_from_index_ = false;
  bool check_filter_;
  // TODO(Zhongyi): pick a better name
  bool need_upper_bound_check_;

  bool async_read_in_progress_;

  // If `target` is null, seek to first.
  void SeekImpl(const Slice* target, bool async_prefetch);

  void InitDataBlock();
  void AsyncInitDataBlock(bool is_first_pass);
  bool MaterializeCurrentBlock();
  void FindKeyForward();
  void FindBlockForward();
  void FindKeyBackward();
  void CheckOutOfBound();

  // Check if data block is fully within iterate_upper_bound.
  //
  // Note MyRocks may update iterate bounds between seek. To workaround it,
  // we need to check and update data_block_within_upper_bound_ accordingly.
  void CheckDataBlockWithinUpperBound();

  bool CheckPrefixMayMatch(const Slice& ikey, IterDirection direction) {
    if (need_upper_bound_check_ && direction == IterDirection::kBackward) {
      // Upper bound check isn't sufficient for backward direction to
      // guarantee the same result as total order, so disable prefix
      // check.
      return true;
    }
    if (check_filter_ &&
        !table_->PrefixMayMatch(ikey, read_options_, prefix_extractor_,
                                need_upper_bound_check_, &lookup_context_)) {
      // TODO remember the iterator is invalidated because of prefix
      // match. This can avoid the upper level file iterator to falsely
      // believe the position is the end of the SST file and move to
      // the first key of the next file.
      ResetDataIter();
      return false;
    }
    return true;
  }
};

#ifdef SMARTSSD

class NDPIterator final : public InternalIterator {
 public:
  NDPIterator(void* es_ndpo, int id, const ReadOptions& read_options, 
      uint16_t key_size) {
    ndp_iter_.Initialize(es_ndpo, id, key_size);
  }

  ~NDPIterator() override {}

  // An iterator is either positioned at a key/value pair, or
  // not valid.  This method returns true iff the iterator is valid.
  // Always returns false if !status().ok().
  bool Valid() const override { return ndp_iter_.Valid(); }

  // Position at the first key in the source.  The iterator is Valid()
  // after this call iff the source is not empty.
  void SeekToFirst() {
    ndp_iter_.SeekToFirst();
  }

  // Position at the last key in the source.  The iterator is
  // Valid() after this call iff the source is not empty.
  void SeekToLast() {
    ndp_iter_.SeekToLast();
  }

  // Position at the first key in the source that at or past target
  // The iterator is Valid() after this call iff the source contains
  // an entry that comes at or past target.
  // All Seek*() methods clear any error status() that the iterator had prior to
  // the call; after the seek, status() indicates only the error (if any) that
  // happened during the seek, not any past errors.
  // 'target' contains user timestamp if timestamp is enabled.
  void Seek(const Slice& target) {
    ndp_iter_.Seek(target);
  }

  // Position at the first key in the source that at or before target
  // The iterator is Valid() after this call iff the source contains
  // an entry that comes at or before target.
  void SeekForPrev(const Slice& target) {
    ndp_iter_.Seek(target);
  }

  // Moves to the next entry in the source.  After this call, Valid() is
  // true iff the iterator was not positioned at the last entry in the source.
  // REQUIRES: Valid()
  void Next() {
    ndp_iter_.Next();
  }

  // Moves to the next entry in the source, and return result. Iterator
  // implementation should override this method to help methods inline better,
  // or when UpperBoundCheckResult() is non-trivial.
  // REQUIRES: Valid()
  bool NextAndGetResult(IterateResult* result) {
    return ndp_iter_.NextAndGetResult(result);
  }

  // Moves to the previous entry in the source.  After this call, Valid() is
  // true iff the iterator was not positioned at the first entry in source.
  // REQUIRES: Valid()
  void Prev() {
    ndp_iter_.Prev();
  }

  // Return the key for the current entry.  The underlying storage for
  // the returned slice is valid only until the next modification of
  // the iterator.
  // REQUIRES: Valid()
  Slice key() const override {
    assert(Valid());
    return ndp_iter_.key();
  }

  // Return the value for the current entry.  The underlying storage for
  // the returned slice is valid only until the next modification of
  // the iterator.
  // REQUIRES: Valid()
  // REQUIRES: PrepareValue() has been called if needed (see PrepareValue()).
  Slice value() const override {
    assert(Valid());
    return ndp_iter_.value();
  }

  // If an error has occurred, return it.  Else return an ok status.
  // If non-blocking IO is requested and this operation cannot be
  // satisfied without doing some IO, then this returns Status::Incomplete().
  Status status() const override {
    return ndp_iter_.status();
  } 

  // Return user key for the current entry.
  // REQUIRES: Valid()
  Slice user_key() const { 
    return ndp_iter_.user_key();
  }

  bool PrepareValue() { 
    return ndp_iter_.PrepareValue(); 
  }

  // Keys return from this iterator can be smaller than iterate_lower_bound.
  bool MayBeOutOfLowerBound() { 
    return ndp_iter_.MayBeOutOfLowerBound();
  }

  // If the iterator has checked the key against iterate_upper_bound, returns
  // the result here. The function can be used by user of the iterator to skip
  // their own checks. If Valid() = true, IterBoundCheck::kUnknown is always
  // a valid value. If Valid() = false, IterBoundCheck::kOutOfBound indicates
  // that the iterator is filtered out by upper bound checks.
  IterBoundCheck UpperBoundCheckResult() {
    return ndp_iter_.UpperBoundCheckResult();
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
  bool IsKeyPinned() const { 
    return ndp_iter_.IsKeyPinned(); 
  }

  // If true, this means that the Slice returned by value() is valid as long as
  // PinnedIteratorsManager::ReleasePinnedData is not called and the
  // Iterator is not deleted.
  // REQUIRES: Same as for value().
  bool IsValuePinned() const { 
    return ndp_iter_.IsValuePinned(); 
  }

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

 private:
  NDPResultsIter ndp_iter_;
};

#endif /* SMARTSSD */

}  // namespace ROCKSDB_NAMESPACE
