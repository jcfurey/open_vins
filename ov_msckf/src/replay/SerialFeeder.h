/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef OV_MSCKF_SERIAL_FEEDER_H
#define OV_MSCKF_SERIAL_FEEDER_H

#include <cstddef>
#include <deque>
#include <functional>
#include <map>
#include <memory>

#include "utils/sensor_data.h"

namespace ov_msckf {

class VioManager;
struct VioManagerOptions;

/**
 * @brief Make estimator settings reproducible for offline replay.
 *
 * Initialization runs synchronously, and the dynamic initializer's optimization runs on one thread and ends on its
 * iteration limit instead of a wall-clock budget (both the budget and Ceres' threaded cost sums vary between runs).
 * @return true if the dynamic initializer settings were changed
 */
bool make_replay_deterministic(VioManagerOptions &params);

/**
 * @brief Deterministic offline driver for the estimator.
 *
 * Measurements are consumed on the calling thread in the order given. Camera packets wait until an IMU sample covers
 * their timestamp plus the current camera-to-IMU offset, which is the same release rule as the live ROS 2 worker.
 * Nothing is dropped for load, so a fixed input always produces the same estimator call sequence.
 */
class SerialFeeder {
public:
  struct Statistics {
    size_t imu_accepted = 0, imu_rejected = 0;
    size_t cameras_accepted = 0, cameras_rejected = 0, cameras_rate_limited = 0;
    size_t cameras_processed = 0, cameras_unprocessed = 0;
    double processing_seconds = 0.0; ///< Wall time spent inside VioManager::feed_measurement_camera
  };

  /// Called after each processed camera packet with the wall time the estimator spent on it.
  using UpdateCallback = std::function<void(const ov_core::CameraData &packet, double processing_seconds)>;

  explicit SerialFeeder(std::shared_ptr<VioManager> app, UpdateCallback on_update = {});

  /// Feed an IMU sample. Non-finite or non-increasing samples are rejected and return false.
  bool feed_imu(const ov_core::ImuData &imu);

  /// Queue a synchronized camera packet. Packets must be time ordered per camera; out-of-order or
  /// rate-limited packets (closer than 1 / track_frequency) are rejected and return false.
  bool feed_camera(ov_core::CameraData packet);

  /// Signal end of input. Packets that IMU data never covered are counted as unprocessed.
  void finish();

  const Statistics &statistics() const { return stats_; }
  size_t pending() const { return pending_.size(); }

private:
  void drain();

  std::shared_ptr<VioManager> app_;
  UpdateCallback on_update_;
  double min_camera_dt_;
  double last_imu_time_;
  std::map<int, double> last_camera_times_;
  std::deque<ov_core::CameraData> pending_;
  Statistics stats_;
};

} // namespace ov_msckf

#endif // OV_MSCKF_SERIAL_FEEDER_H
