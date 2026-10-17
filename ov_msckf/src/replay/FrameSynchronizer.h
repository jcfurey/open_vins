/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef OV_MSCKF_FRAME_SYNCHRONIZER_H
#define OV_MSCKF_FRAME_SYNCHRONIZER_H

#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace ov_msckf {

/**
 * @brief Groups independently recorded camera streams into synchronized frames, deterministically.
 *
 * Camera 0 is the reference. A frame is emitted once every other camera has an image within the tolerance of the
 * reference stamp; reference or partner images that can no longer be matched are discarded and counted.
 */
class FrameSynchronizer {
public:
  struct Frame {
    double timestamp; ///< Reference (camera 0) stamp
    std::vector<cv::Mat> images;
  };

  FrameSynchronizer(size_t num_cameras, double tolerance)
      : queues_(num_cameras), last_(num_cameras, -std::numeric_limits<double>::infinity()), tolerance_(tolerance) {
    if (num_cameras == 0 || !std::isfinite(tolerance) || tolerance < 0)
      throw std::invalid_argument("FrameSynchronizer needs cameras and a finite, non-negative tolerance");
  }

  /// Add an image; returns false for an unknown camera or a stamp that does not increase.
  bool push(size_t camera, double timestamp, cv::Mat image) {
    if (camera >= queues_.size() || !std::isfinite(timestamp) || timestamp <= last_[camera])
      return false;
    last_[camera] = timestamp;
    queues_[camera].emplace_back(timestamp, std::move(image));
    return true;
  }

  /// Pop the next synchronized frame if one is complete.
  bool pop(Frame &frame) {
    while (!queues_[0].empty()) {
      const double reference = queues_[0].front().first;
      bool complete = true, reference_unmatched = false;
      for (size_t id = 1; id < queues_.size(); id++) {
        auto &queue = queues_[id];
        while (!queue.empty() && queue.front().first < reference - tolerance_) {
          queue.pop_front();
          unmatched_++;
        }
        if (queue.empty())
          complete = false;
        else if (queue.front().first > reference + tolerance_)
          reference_unmatched = true;
      }
      if (reference_unmatched) {
        queues_[0].pop_front();
        unmatched_++;
        continue;
      }
      if (!complete)
        return false;
      frame.timestamp = reference;
      frame.images.clear();
      for (auto &queue : queues_) {
        frame.images.push_back(std::move(queue.front().second));
        queue.pop_front();
      }
      return true;
    }
    return false;
  }

  /// Images that were (or, after the input ends, still are) without a complete frame.
  size_t unmatched() const {
    size_t pending = 0;
    for (const auto &queue : queues_)
      pending += queue.size();
    return unmatched_ + pending;
  }

private:
  std::vector<std::deque<std::pair<double, cv::Mat>>> queues_;
  std::vector<double> last_;
  double tolerance_;
  size_t unmatched_ = 0;
};

} // namespace ov_msckf

#endif // OV_MSCKF_FRAME_SYNCHRONIZER_H
