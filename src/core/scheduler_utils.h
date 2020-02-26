// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <deque>
#include <unordered_map>
#include "src/core/model_config.h"
#include "src/core/scheduler.h"
#include "src/core/server_status.h"

namespace nvidia { namespace inferenceserver {

using PendingBatchShapes =
    std::unordered_map<std::string, std::pair<DimsList, std::vector<int64_t>>>;

Status InitPendingShape(
    const int64_t runner_id, const Scheduler::Payload& payload,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const Scheduler::StandardShapeTensorPeekFunc& OnPeek,
    PendingBatchShapes* pending_batch_shapes);

bool CompareWithPendingShape(
    const int64_t runner_id, const Scheduler::Payload& payload,
    const Scheduler::StandardShapeTensorPeekFunc& OnPeek,
    const PendingBatchShapes& pending_batch_shapes);

using ModelQueuePolicyMap =
    ::google::protobuf::Map<::google::protobuf::uint32, ModelQueuePolicy>;

class RequestQueue {
 public:
  RequestQueue()
      : timeout_action_(ModelQueuePolicy::REJECT),
        default_timeout_microseconds_(0), allow_timeout_override_(false),
        max_queue_size_(0)
  {
  }

  RequestQueue(const ModelQueuePolicy& policy)
      : timeout_action_(policy.timeout_action()),
        default_timeout_microseconds_(policy.default_timeout_microseconds()),
        allow_timeout_override_(policy.allow_timeout_override()),
        max_queue_size_(policy.max_queue_size())
  {
  }

  // Enqueue an payload and set up its timeout accordingly.
  Status Enqueue(Scheduler::Payload&& payload);

  // Dequeue the payload at the front of the queue.
  Scheduler::Payload Dequeue();

  // Apply the queue policy to payload at 'idx'.
  // Return true if the 'idx' still points to an payload after applying the
  // policy, false otherwise.
  bool ApplyPolicy(size_t idx);

  // Return the rejected payloads held by the request queue.
  std::deque<Scheduler::Payload> ReleaseRejectedQueue();

  // Return the payload at 'idx'.
  Scheduler::Payload& At(size_t idx);

  // Return the timeout timestamp of the payload at 'idx', in ns. A value of 0
  // indicates that the payload doesn't specify a timeout.
  uint64_t TimeoutAt(size_t idx);

  bool Empty() { return Size() == 0; }

  size_t Size() { return queue_.size() + delayed_queue_.size(); }

 private:
  std::deque<Scheduler::Payload> queue_;
  std::deque<uint64_t> timeout_timestamp_ns_;
  std::deque<Scheduler::Payload> delayed_queue_;
  std::deque<Scheduler::Payload> rejected_queue_;
  const ModelQueuePolicy::TimeoutAction timeout_action_;
  const uint64_t default_timeout_microseconds_;
  const bool allow_timeout_override_;
  const uint32_t max_queue_size_;
};

class PriorityQueue {
 public:
  PriorityQueue();

  PriorityQueue(
      const ModelQueuePolicy& default_queue_policy, uint32_t priority_levels,
      const ModelQueuePolicyMap queue_policy_map);

  Status Enqueue(uint32_t priority_level, Scheduler::Payload&& payload);

  Scheduler::Payload Dequeue();

  size_t Size();

  bool Empty() { return Size() == 0; }

  Scheduler::Payload& PayloadAtCursor() { return pending_cursor_.GetItem(); }

  void MarkCursor() { current_mark_ = pending_cursor_; }

  void AdvanceCursor() { pending_cursor_.Next(); }

  bool CursorEnd()
  {
    return pending_cursor_.curr_it_ == pending_cursor_.end_it_;
  }

  void ResetCursor()
  {
    pending_cursor_ = Cursor(queues_.begin(), queues_.end());
  }

  void SetCursorToMark() { pending_cursor_ = current_mark_; }

  bool IsCursorValid();

  uint64_t OldestEnqueueTime()
  {
    return pending_cursor_.pending_batch_oldest_enqueue_time_ns_;
  }

 private:
  using PriorityQueues = std::map<uint32_t, RequestQueue>;
  PriorityQueues queues_;

  // Cursor which points to the item after the pending batch
  struct Cursor {
    Cursor() = default;
    Cursor(PriorityQueues::iterator start_it, PriorityQueues::iterator end_it);

    Cursor(const Cursor& rhs)
        : curr_it_(rhs.curr_it_), end_it_(rhs.end_it_),
          queue_idx_(rhs.queue_idx_),
          pending_batch_closest_timeout_ns_(
              rhs.pending_batch_closest_timeout_ns_),
          pending_batch_oldest_enqueue_time_ns_(
              rhs.pending_batch_oldest_enqueue_time_ns_),
          valid_(rhs.valid_)
    {
    }

    Scheduler::Payload& GetItem() { return curr_it_->second.At(queue_idx_); }

    void Next();

    PriorityQueues::iterator curr_it_;
    PriorityQueues::iterator end_it_;
    size_t queue_idx_;
    uint64_t pending_batch_closest_timeout_ns_;
    uint64_t pending_batch_oldest_enqueue_time_ns_;
    bool valid_;
  };

  Cursor pending_cursor_;
  Cursor current_mark_;
};

}}  // namespace nvidia::inferenceserver
