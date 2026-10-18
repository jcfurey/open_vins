/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_MSCKF_ROS2_MEASUREMENT_BUFFER_H
#define OV_MSCKF_ROS2_MEASUREMENT_BUFFER_H

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

#include "utils/sensor_data.h"

namespace ov_msckf {

/// Bounded sensor ingress. Only the consumer accesses estimator state.
class Ros2MeasurementBuffer {
public:
  enum class Result { Accepted, DroppedOldest, Invalid, OutOfOrder, RateLimited, Reset, Stopped };
  struct Batch {
    std::deque<ov_core::ImuData> imu;
    std::optional<ov_core::CameraData> camera;
    bool reset = false;
  };
  struct Statistics {
    uint64_t imu_accepted = 0, cameras_accepted = 0, imu_dropped = 0, cameras_dropped = 0;
    uint64_t invalid = 0, out_of_order = 0, rate_limited = 0, resets = 0;
    size_t imu_pending = 0, cameras_pending = 0, imu_peak = 0, camera_peak = 0;
  };

  Ros2MeasurementBuffer(size_t imu_limit, size_t camera_limit, size_t camera_count, double tracking_rate, bool reset_on_time_jump = true,
                       double max_camera_gap = 1.0)
      : imu_limit_(imu_limit), camera_limit_(camera_limit), minimum_camera_dt_(1.0 / tracking_rate),
        max_camera_gap_(max_camera_gap), reset_on_time_jump_(reset_on_time_jump),
        camera_times_(camera_count, -std::numeric_limits<double>::infinity()) {
    if (!imu_limit || !camera_limit || !camera_count || !std::isfinite(tracking_rate) || tracking_rate <= 0 ||
        !std::isfinite(max_camera_gap) || max_camera_gap <= 0)
      throw std::invalid_argument("Sensor queue sizes, camera count, tracking rate, and maximum camera gap must be positive");
  }

  Result push_imu(const ov_core::ImuData &message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_)
      return Result::Stopped;
    if (!std::isfinite(message.timestamp) || message.timestamp < 0 || !message.wm.allFinite() || !message.am.allFinite()) {
      ++statistics_.invalid;
      return Result::Invalid;
    }
    Result result = Result::Accepted;
    if (message.timestamp <= latest_imu_time_) {
      if (reset_on_time_jump_ && latest_imu_time_ - message.timestamp > 1.0) {
        reset_locked();
        result = Result::Reset;
      } else {
        ++statistics_.out_of_order;
        return Result::OutOfOrder;
      }
    }
    latest_imu_time_ = message.timestamp;
    if (imu_.size() == imu_limit_) {
      imu_.pop_front();
      ++statistics_.imu_dropped;
      statistics_.cameras_dropped += cameras_.size();
      cameras_.clear();
      if (!reset_pending_)
        ++statistics_.resets;
      reset_pending_ = true;
      result = Result::Reset;
    }
    imu_.push_back(message);
    ++statistics_.imu_accepted;
    statistics_.imu_peak = std::max(statistics_.imu_peak, imu_.size());
    condition_.notify_one();
    return result;
  }

  Result push_camera(ov_core::CameraData message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_)
      return Result::Stopped;
    if (!std::isfinite(message.timestamp) || message.timestamp < 0 || message.sensor_ids.empty() ||
        message.images.size() != message.sensor_ids.size() || message.masks.size() != message.images.size()) {
      ++statistics_.invalid;
      return Result::Invalid;
    }
    std::vector<bool> seen(camera_times_.size(), false);
    for (size_t i = 0; i < message.sensor_ids.size(); ++i) {
      int id = message.sensor_ids[i];
      if (id < 0 || static_cast<size_t>(id) >= seen.size() || seen[id] || message.images[i].empty() ||
          message.images[i].type() != CV_8UC1 || message.masks[i].type() != CV_8UC1 ||
          message.images[i].size() != message.masks[i].size()) {
        ++statistics_.invalid;
        return Result::Invalid;
      }
      seen[id] = true;
      if (message.timestamp <= camera_times_[id]) {
        ++statistics_.out_of_order;
        return Result::OutOfOrder;
      }
      // Nanosecond stamp conversion must not reject frames exactly at the configured rate.
      if (message.timestamp - camera_times_[id] < minimum_camera_dt_ - 1e-6) {
        ++statistics_.rate_limited;
        return Result::RateLimited;
      }
    }
    for (int id : message.sensor_ids)
      camera_times_[id] = message.timestamp;
    auto position = std::upper_bound(cameras_.begin(), cameras_.end(), message);
    cameras_.insert(position, std::move(message));
    Result result = Result::Accepted;
    if (cameras_.size() > camera_limit_) {
      cameras_.pop_front();
      ++statistics_.cameras_dropped;
      // Lost visual updates invalidate the old track/state history under overload.
      // Reinitialize with bounded, recent input instead of silently propagating through the loss.
      if (!reset_pending_)
        ++statistics_.resets;
      reset_pending_ = true;
      result = Result::DroppedOldest;
    }
    ++statistics_.cameras_accepted;
    statistics_.camera_peak = std::max(statistics_.camera_peak, cameras_.size());
    condition_.notify_one();
    return result;
  }

  /// Wait for IMU input or a frame bracketed by IMU. Estimation never holds this lock.
  bool wait_take(Batch &batch, double fed_imu_time, double camera_to_imu_offset) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return stopped_ || reset_pending_ || !imu_.empty() || camera_ready(fed_imu_time, camera_to_imu_offset); });
    if (stopped_)
      return false;
    if (!reset_pending_ && !cameras_.empty() && cameras_.front().timestamp - last_camera_time_ > max_camera_gap_ &&
        std::isfinite(last_camera_time_)) {
      reset_pending_ = true;
      ++statistics_.resets;
    }
    batch = Batch{};
    batch.reset = reset_pending_;
    reset_pending_ = false;
    if (batch.reset)
      last_camera_time_ = -std::numeric_limits<double>::infinity();
    // Hand over a frame together with only the IMU that reaches it, so the estimator sees measurements in the
    // same order as an offline replay no matter how far behind the worker is.
    const double fed = batch.reset ? -1.0 : fed_imu_time;
    if (!cameras_.empty() && std::isfinite(camera_to_imu_offset)) {
      const double needed = cameras_.front().timestamp + camera_to_imu_offset;
      const auto cover =
          std::lower_bound(imu_.begin(), imu_.end(), needed, [](const ov_core::ImuData &imu, double time) { return imu.timestamp < time; });
      if (fed >= needed || cover != imu_.end()) {
        const auto end = fed >= needed ? imu_.begin() : std::next(cover);
        batch.imu.assign(std::make_move_iterator(imu_.begin()), std::make_move_iterator(end));
        imu_.erase(imu_.begin(), end);
        batch.camera = std::move(cameras_.front());
        cameras_.pop_front();
        last_camera_time_ = batch.camera->timestamp;
        return true;
      }
    }
    batch.imu.swap(imu_);
    return true;
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!stopped_) {
      reset_locked();
      condition_.notify_one();
    }
  }

  void stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
    imu_.clear();
    cameras_.clear();
    condition_.notify_all();
  }

  Statistics statistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = statistics_;
    result.imu_pending = imu_.size();
    result.cameras_pending = cameras_.size();
    return result;
  }

private:
  bool camera_ready(double imu_time, double offset) const {
    return !cameras_.empty() && std::isfinite(offset) && imu_time >= cameras_.front().timestamp + offset;
  }
  void reset_locked() {
    statistics_.imu_dropped += imu_.size();
    statistics_.cameras_dropped += cameras_.size();
    imu_.clear();
    cameras_.clear();
    std::fill(camera_times_.begin(), camera_times_.end(), -std::numeric_limits<double>::infinity());
    latest_imu_time_ = -std::numeric_limits<double>::infinity();
    last_camera_time_ = -std::numeric_limits<double>::infinity();
    reset_pending_ = true;
    ++statistics_.resets;
  }

  const size_t imu_limit_, camera_limit_;
  const double minimum_camera_dt_, max_camera_gap_;
  const bool reset_on_time_jump_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  bool stopped_ = false, reset_pending_ = false;
  double latest_imu_time_ = -std::numeric_limits<double>::infinity();
  double last_camera_time_ = -std::numeric_limits<double>::infinity();
  std::vector<double> camera_times_;
  std::deque<ov_core::ImuData> imu_;
  std::deque<ov_core::CameraData> cameras_;
  Statistics statistics_;
};

} // namespace ov_msckf
#endif
