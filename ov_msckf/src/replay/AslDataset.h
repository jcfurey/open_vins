/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef OV_MSCKF_ASL_DATASET_H
#define OV_MSCKF_ASL_DATASET_H

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "utils/sensor_data.h"

namespace ov_msckf {

/// Convert integer nanoseconds exactly as ROS stamps are converted (seconds + nanoseconds * 1e-9).
double asl_seconds(int64_t nanoseconds);

/**
 * @brief Index of an ASL ("EuRoC format") dataset: `mav0/imu0/data.csv` and `mav0/camN/data.csv` + `data/` images.
 *
 * Used by EuRoC MAV and the TUM-VI EuRoC exports. Camera frames are grouped across cameras by identical
 * timestamps (within a tolerance); unmatched frames are counted and skipped.
 */
class AslDataset {
public:
  struct Frame {
    int64_t timestamp_ns;
    std::vector<std::string> paths; ///< One image path per camera, index = camera id
  };

  /// @param root Either the sequence folder containing `mav0/` or the `mav0/` folder itself
  AslDataset(const std::string &root, int num_cameras, double sync_tolerance_s = 0.001);

  const std::vector<ov_core::ImuData> &imu() const { return imu_; }
  const std::vector<int64_t> &imu_timestamps_ns() const { return imu_ns_; }
  const std::vector<Frame> &frames() const { return frames_; }
  size_t unmatched_frames() const { return unmatched_; }
  const std::string &folder() const { return folder_; }

private:
  std::string folder_;
  std::vector<ov_core::ImuData> imu_;
  std::vector<int64_t> imu_ns_;
  std::vector<Frame> frames_;
  size_t unmatched_ = 0;
};

/**
 * @brief Decodes dataset images in order on a background thread so the estimator thread never waits on PNG decoding.
 *
 * Frames are delivered strictly in the given order, so prefetching never changes results.
 */
class ImagePrefetcher {
public:
  ImagePrefetcher(std::vector<AslDataset::Frame> frames, size_t capacity);
  ~ImagePrefetcher();
  ImagePrefetcher(const ImagePrefetcher &) = delete;
  ImagePrefetcher &operator=(const ImagePrefetcher &) = delete;

  /// Blocks for the next frame. Returns false after the last frame; rethrows decoding errors.
  bool next(int64_t &timestamp_ns, std::vector<cv::Mat> &images);

private:
  void run();

  std::vector<AslDataset::Frame> frames_;
  size_t capacity_, consumed_ = 0;
  std::deque<std::vector<cv::Mat>> ready_;
  std::exception_ptr error_;
  bool stop_ = false, done_ = false;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::thread worker_;
};

} // namespace ov_msckf

#endif // OV_MSCKF_ASL_DATASET_H
