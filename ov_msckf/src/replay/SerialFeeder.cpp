/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "SerialFeeder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "core/VioManager.h"
#include "state/State.h"
#include "types/Vec.h"

using namespace ov_msckf;

SerialFeeder::SerialFeeder(std::shared_ptr<VioManager> app, UpdateCallback on_update)
    : app_(std::move(app)), on_update_(std::move(on_update)), last_imu_time_(-std::numeric_limits<double>::infinity()) {
  if (app_ == nullptr)
    throw std::invalid_argument("SerialFeeder requires an estimator");
  const double rate = app_->get_params().track_frequency;
  if (!std::isfinite(rate) || rate <= 0)
    throw std::invalid_argument("track_frequency must be finite and positive");
  min_camera_dt_ = 1.0 / rate;
}

bool SerialFeeder::feed_imu(const ov_core::ImuData &imu) {
  if (!std::isfinite(imu.timestamp) || imu.timestamp < 0 || imu.timestamp <= last_imu_time_ || !imu.wm.allFinite() || !imu.am.allFinite()) {
    ++stats_.imu_rejected;
    return false;
  }
  last_imu_time_ = imu.timestamp;
  ++stats_.imu_accepted;
  app_->feed_measurement_imu(imu);
  drain();
  return true;
}

bool SerialFeeder::feed_camera(ov_core::CameraData packet) {
  bool valid = std::isfinite(packet.timestamp) && packet.timestamp >= 0 && !packet.sensor_ids.empty() &&
               packet.images.size() == packet.sensor_ids.size() && packet.masks.size() == packet.images.size();
  for (size_t i = 0; valid && i < packet.sensor_ids.size(); i++) {
    const cv::Mat &image = packet.images.at(i), &mask = packet.masks.at(i);
    valid = !image.empty() && image.type() == CV_8UC1 && mask.type() == CV_8UC1 && mask.size() == image.size() &&
            std::count(packet.sensor_ids.begin(), packet.sensor_ids.end(), packet.sensor_ids.at(i)) == 1;
    const auto last = last_camera_times_.find(packet.sensor_ids.at(i));
    valid = valid && (last == last_camera_times_.end() || packet.timestamp > last->second);
  }
  if (!valid) {
    ++stats_.cameras_rejected;
    return false;
  }
  // Nanosecond stamp conversion must not reject frames exactly at the configured rate (same rule as the ROS 2 node).
  for (int id : packet.sensor_ids) {
    const auto last = last_camera_times_.find(id);
    if (last != last_camera_times_.end() && packet.timestamp - last->second < min_camera_dt_ - 1e-6) {
      ++stats_.cameras_rate_limited;
      return false;
    }
  }
  for (int id : packet.sensor_ids)
    last_camera_times_[id] = packet.timestamp;
  ++stats_.cameras_accepted;
  pending_.insert(std::upper_bound(pending_.begin(), pending_.end(), packet), std::move(packet));
  drain();
  return true;
}

void SerialFeeder::finish() {
  drain();
  stats_.cameras_unprocessed += pending_.size();
  pending_.clear();
}

void SerialFeeder::drain() {
  while (!pending_.empty()) {
    // Use the current estimate of the time offset, as the live node does.
    const double offset = app_->get_state()->_calib_dt_CAMtoIMU->value()(0);
    if (!(last_imu_time_ >= pending_.front().timestamp + offset))
      return;
    ov_core::CameraData packet = std::move(pending_.front());
    pending_.pop_front();
    const auto start = std::chrono::steady_clock::now();
    app_->feed_measurement_camera(packet);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ++stats_.cameras_processed;
    stats_.processing_seconds += seconds;
    if (on_update_)
      on_update_(packet, seconds);
  }
}
