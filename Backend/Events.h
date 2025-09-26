// src/Backend/Events.h
#pragma once
#ifndef __METAL_VERSION__

#include "Resource.h"
#include <memory>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
#endif

#ifdef USE_METAL
#include "Backend/METAL/METALManager.h"
#endif

namespace ARBD {

class Event {
private:
  Resource resource_;
  std::shared_ptr<void> event_impl_; // Type-erased backend event

public:
  Event() = default;

  // Null constructor for cases where no event is needed
  Event(std::nullptr_t, const Resource &res)
      : resource_(res), event_impl_(nullptr) {}

#ifdef USE_CUDA
  Event(cudaEvent_t cuda_event, const Resource &res)
      : resource_(res), event_impl_(new cudaEvent_t(cuda_event), [](void *p) {
          cudaEvent_t *evt = static_cast<cudaEvent_t *>(p);
          if (evt && *evt) {
            cudaEventDestroy(*evt);
          }
          delete evt;
        }) {
    // Validate resource type matches
    if (res.type() != ResourceType::CUDA) {
      ARBD_Exception(ExceptionType::ValueError,
                     "CUDA event requires CUDA resource, got {}",
                     res.getTypeString());
    }
  }
#endif

#ifdef USE_SYCL
  Event(sycl::event sycl_event, const Resource &res)
      : resource_(res),
        event_impl_(std::make_shared<sycl::event>(std::move(sycl_event))) {
    // Validate resource type matches
    if (res.type() != ResourceType::SYCL) {
      ARBD_Exception(ExceptionType::ValueError,
                     "SYCL event requires SYCL resource, got {}",
                     res.getTypeString());
    }
  }
#endif

#ifdef USE_METAL
  Event(ARBD::METAL::Event metal_event, const Resource &res)
      : resource_(res), event_impl_(std::make_shared<ARBD::METAL::Event>(
                            std::move(metal_event))) {
    // Validate resource type matches
    if (res.type() != ResourceType::METAL) {
      ARBD_Exception(ExceptionType::ValueError,
                     "Metal event requires Metal resource, got {}",
                     res.getTypeString());
    }
  }
#endif

  void wait() const {
    if (!event_impl_)
      return;

    // Use resource type to dispatch to correct backend
    switch (resource_.type()) {
#ifdef USE_CUDA
    case ResourceType::CUDA: {
      if (auto *cuda_event_ptr =
              static_cast<cudaEvent_t *>(event_impl_.get())) {
        cudaEventSynchronize(*cuda_event_ptr);
      }
      return;
    }
#endif
#ifdef USE_SYCL
    case ResourceType::SYCL: {
      if (auto *sycl_event = static_cast<sycl::event *>(event_impl_.get())) {
        sycl_event->wait();
      }
      return;
    }
#endif
#ifdef USE_METAL
    case ResourceType::METAL: {
      if (auto *metal_event =
              static_cast<ARBD::METAL::Event *>(event_impl_.get())) {
        metal_event->wait();
      }
      return;
    }
#endif
    case ResourceType::CPU:
    default:
      // CPU events don't need waiting
      return;
    }
  }

  bool is_complete() const {
    if (!event_impl_)
      return true;

    switch (resource_.type()) {
#ifdef USE_CUDA
    case ResourceType::CUDA: {
      if (auto *cuda_event = static_cast<cudaEvent_t *>(event_impl_.get())) {
        cudaError_t status = cudaEventQuery(*cuda_event);
        return status == cudaSuccess;
      }
      return true;
    }
#endif
#ifdef USE_SYCL
    case ResourceType::SYCL: {
      if (auto *sycl_event = static_cast<sycl::event *>(event_impl_.get())) {
        return sycl_event
                   ->get_info<sycl::info::event::command_execution_status>() ==
               sycl::info::event_command_status::complete;
      }
      return true;
    }
#endif
#ifdef USE_METAL
    case ResourceType::METAL: {
      if (auto *metal_event =
              static_cast<ARBD::METAL::Event *>(event_impl_.get())) {
        return metal_event->is_complete();
      }
      return true;
    }
#endif
    case ResourceType::CPU:
    default:
      return true;
    }
  }

  const Resource &get_resource() const { return resource_; }
  bool is_valid() const { return event_impl_ != nullptr; }
  void *get_event_impl() const { return event_impl_.get(); }
};

// EventList remains the same - no changes needed
class EventList {
private:
  std::vector<Event> events_;

public:
  EventList() = default;
  EventList(std::initializer_list<Event> events) : events_(events) {}

  void add(const Event &event) {
    if (event.is_valid()) {
      events_.push_back(event);
    }
  }

  void wait_all() const {
    for (const auto &event : events_) {
      event.wait();
    }
  }

  bool all_complete() const {
    for (const auto &event : events_) {
      if (!event.is_complete())
        return false;
    }
    return true;
  }

  const std::vector<Event> &get_events() const { return events_; }
  bool empty() const { return events_.empty(); }
  size_t size() const { return events_.size(); }
  void clear() { events_.clear(); }

#ifdef USE_CUDA
  std::vector<cudaEvent_t> get_cuda_events() const {
    std::vector<cudaEvent_t> cuda_events;
    cuda_events.reserve(events_.size());
    for (const auto &event : events_) {
      if (event.is_valid() &&
          event.get_resource().type() == ResourceType::CUDA) {
        if (auto *impl = static_cast<cudaEvent_t *>(event.get_event_impl())) {
          cuda_events.push_back(*impl);
        }
      }
    }
    return cuda_events;
  }
#endif

#ifdef USE_SYCL
  std::vector<sycl::event> get_sycl_events() const {
    std::vector<sycl::event> sycl_events;
    sycl_events.reserve(events_.size());
    for (const auto &event : events_) {
      if (event.is_valid() &&
          event.get_resource().type() == ResourceType::SYCL) {
        if (auto *impl = static_cast<sycl::event *>(event.get_event_impl())) {
          sycl_events.push_back(*impl);
        }
      }
    }
    return sycl_events;
  }
#endif

#ifdef USE_METAL
  std::vector<ARBD::METAL::Event *> get_metal_events() const {
    std::vector<ARBD::METAL::Event *> metal_events;
    metal_events.reserve(events_.size());
    for (const auto &event : events_) {
      if (event.is_valid() &&
          event.get_resource().type() == ResourceType::METAL) {
        if (auto *impl =
                static_cast<ARBD::METAL::Event *>(event.get_event_impl())) {
          metal_events.push_back(impl);
        }
      }
    }
    return metal_events;
  }
#endif
};

} // namespace ARBD
#endif // __METAL_VERSION__
