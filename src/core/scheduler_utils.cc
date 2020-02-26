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

#include "src/core/scheduler_utils.h"

#include "src/core/constants.h"
#include "src/core/provider.h"

namespace nvidia { namespace inferenceserver {

Status
InitPendingShape(
    const int64_t runner_id, const Scheduler::Payload& payload,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const Scheduler::StandardShapeTensorPeekFunc& OnPeek,
    PendingBatchShapes* pending_batch_shapes)
{
  pending_batch_shapes->clear();

  const InferRequestHeader& request =
      payload.request_provider_->RequestHeader();
  for (const auto& input : request.input()) {
    const auto itr = enforce_equal_shape_tensors.find(input.name());
    if (itr != enforce_equal_shape_tensors.end()) {
      std::pair<DimsList, std::vector<int64_t>> shapes;
      shapes.first = input.dims();

      // For shape tensors must compare the contents of the tensor in
      // addition to the tensor shape itself.
      if (itr->second) {
        RETURN_IF_ERROR(OnPeek(runner_id, input, payload, &shapes.second));
      }

      pending_batch_shapes->emplace(
          std::make_pair(input.name(), std::move(shapes)));
    }
  }

  return Status::Success;
}

bool
CompareWithPendingShape(
    const int64_t runner_id, const Scheduler::Payload& payload,
    const Scheduler::StandardShapeTensorPeekFunc& OnPeek,
    const PendingBatchShapes& pending_batch_shapes)
{
  const InferRequestHeader& request =
      payload.request_provider_->RequestHeader();

  for (const auto& input : request.input()) {
    const auto itr = pending_batch_shapes.find(input.name());
    if (itr != pending_batch_shapes.end()) {
      if (!CompareDims(itr->second.first, input.dims())) {
        return false;
      }

      // If there are shape-tensor contents then compare those as
      // well.
      if (!itr->second.second.empty()) {
        std::vector<int64_t> shape;

        // If fail getting the tensor shape then conservatively return
        // false to indicate that the shapes don't match.
        if (!OnPeek(runner_id, input, payload, &shape).IsOk()) {
          return false;
        }
        if (!CompareDims(itr->second.second, shape)) {
          return false;
        }
      }
    }
  }

  return true;
}

Status
RequestQueue::Enqueue(Scheduler::Payload&& payload)
{
  if ((max_queue_size_ != 0) && (queue_.size() >= max_queue_size_)) {
    return Status(RequestStatusCode::UNAVAILABLE, "Exceeds maximum queue size");
  }
  queue_.emplace_back(std::move(payload));
  auto timeout_microseconds = default_timeout_microseconds_;
  if (allow_timeout_override_ &&
      queue_.back().request_provider_->RequestHeader().timeout_microseconds() !=
          0) {
    timeout_microseconds =
        queue_.back().request_provider_->RequestHeader().timeout_microseconds();
  }
  if (timeout_microseconds != 0) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    timeout_timestamp_ns_.emplace_back(
        TIMESPEC_TO_NANOS(now) + timeout_microseconds * 1000);
  } else {
    timeout_timestamp_ns_.emplace_back(0);
  }

  return Status::Success;
}

Scheduler::Payload
RequestQueue::Dequeue()
{
  if (!queue_.empty()) {
    auto res = std::move(queue_.front());
    queue_.pop_front();
    timeout_timestamp_ns_.pop_front();
    return res;
  } else {
    auto res = std::move(delayed_queue_.front());
    delayed_queue_.pop_front();
    return res;
  }
}

bool
RequestQueue::ApplyPolicy(size_t idx)
{
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  auto now_nanoseconds = TIMESPEC_TO_NANOS(now);
  while (idx < queue_.size()) {
    if ((timeout_timestamp_ns_[idx] != 0) &&
        (now_nanoseconds > timeout_timestamp_ns_[idx])) {
      if (timeout_action_ == ModelQueuePolicy::DELAY) {
        delayed_queue_.emplace_back(std::move(queue_[idx]));
      } else {
        rejected_queue_.emplace_back(std::move(queue_[idx]));
      }
      // FIXME: erase on deque is linear, should consider to use different data
      // structure instead. List is the first one to think of, but the traversal
      // may not be as efficient due to cache miss.
      queue_.erase(queue_.begin() + idx);
      timeout_timestamp_ns_.erase(timeout_timestamp_ns_.begin() + idx);
    } else {
      // Current idx is pointing to an item with unexpired timeout
      return true;
    }
  }
  // At this point, idx is pointing to an item with expired timeout.
  // If the item is in delayed queue, then return true. Otherwise, false
  // meaning the queue has no item with this 'idx'.
  return ((idx - queue_.size()) < delayed_queue_.size());
}

std::deque<Scheduler::Payload>
RequestQueue::ReleaseRejectedQueue()
{
  std::deque<Scheduler::Payload> res;
  rejected_queue_.swap(res);
  return res;
}

Scheduler::Payload&
RequestQueue::At(size_t idx)
{
  if (idx < queue_.size()) {
    return queue_[idx];
  } else {
    return delayed_queue_[idx - queue_.size()];
  }
}

uint64_t
RequestQueue::TimeoutAt(size_t idx)
{
  if (idx < queue_.size()) {
    return timeout_timestamp_ns_[idx];
  } else {
    return 0;
  }
}

PriorityQueue::PriorityQueue()
{
  ModelQueuePolicy default_policy;
  queues_.emplace(0, RequestQueue(default_policy));
  ResetCursor();
}

PriorityQueue::PriorityQueue(
    const ModelQueuePolicy& default_queue_policy, uint32_t priority_levels,
    const ModelQueuePolicyMap queue_policy_map)
{
  if (priority_levels == 0) {
    queues_.emplace(0, RequestQueue(default_queue_policy));
  } else {
    for (uint32_t level = 1; level <= priority_levels; level++) {
      auto it = queue_policy_map.find(level);
      if (it == queue_policy_map.end()) {
        queues_.emplace(level, RequestQueue(default_queue_policy));
      } else {
        queues_.emplace(level, RequestQueue(it->second));
      }
    }
  }
  ResetCursor();
}

Status
PriorityQueue::Enqueue(uint32_t priority_level, Scheduler::Payload&& payload)
{
  auto status = queues_[priority_level].Enqueue(std::move(payload));
  if (status.IsOk() && pending_cursor_.valid_) {
    pending_cursor_.valid_ &=
        (priority_level > pending_cursor_.curr_it_->first);
  }
  return status;
}

Scheduler::Payload
PriorityQueue::Dequeue()
{
  pending_cursor_.valid_ = false;
  for (auto& queue : queues_) {
    if (!queue.second.Empty()) {
      return queue.second.Dequeue();
    }
  }
  throw std::out_of_range("dequeue on empty queue");
}

size_t
PriorityQueue::Size()
{
  size_t res = 0;
  for (auto& queue : queues_) {
    res += queue.second.Size();
  }
  return res;
}

bool
PriorityQueue::IsCursorValid()
{
  if (pending_cursor_.valid_) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return TIMESPEC_TO_NANOS(now) <
           pending_cursor_.pending_batch_closest_timeout_ns_;
  }
  return false;
}

PriorityQueue::Cursor::Cursor(
    PriorityQueues::iterator start_it, PriorityQueues::iterator end_it)
    : curr_it_(start_it), end_it_(end_it), queue_idx_(0),
      pending_batch_closest_timeout_ns_(0),
      pending_batch_oldest_enqueue_time_ns_(0), valid_(false)
{
  while (curr_it_ != end_it_) {
    if (!(curr_it_->second.ApplyPolicy(queue_idx_))) {
      curr_it_++;
      queue_idx_ = 0;
    } else {
      pending_batch_closest_timeout_ns_ =
          curr_it_->second.TimeoutAt(queue_idx_);
      pending_batch_oldest_enqueue_time_ns_ = TIMESPEC_TO_NANOS(
          curr_it_->second.At(queue_idx_)
              .stats_->Timestamp(ModelInferStats::TimestampKind::kQueueStart));
      valid_ = true;
      break;
    }
  }
}

void
PriorityQueue::Cursor::Next()
{
  const auto& timeout_ns = curr_it_->second.TimeoutAt(queue_idx_);
  if (timeout_ns != 0) {
    if (pending_batch_closest_timeout_ns_ != 0) {
      pending_batch_closest_timeout_ns_ =
          std::min(pending_batch_closest_timeout_ns_, timeout_ns);
    } else {
      pending_batch_closest_timeout_ns_ = timeout_ns;
    }
  }
  pending_batch_oldest_enqueue_time_ns_ = std::min(
      pending_batch_oldest_enqueue_time_ns_,
      TIMESPEC_TO_NANOS(
          curr_it_->second.At(queue_idx_)
              .stats_->Timestamp(ModelInferStats::TimestampKind::kQueueStart)));
  ++queue_idx_;
  while (curr_it_ != end_it_) {
    if (!(curr_it_->second.ApplyPolicy(queue_idx_))) {
      curr_it_++;
      queue_idx_ = 0;
    } else {
      break;
    }
  }
}

}}  // namespace nvidia::inferenceserver
