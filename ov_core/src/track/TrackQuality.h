/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef OV_CORE_TRACK_QUALITY_H
#define OV_CORE_TRACK_QUALITY_H
#include <Eigen/Core>
#include <limits>
#include <opencv2/core.hpp>
#include <vector>

namespace ov_core {
struct FlowQuality {
  double temporal = 0;
  double balance = std::numeric_limits<double>::quiet_NaN();
  double stereo = std::numeric_limits<double>::quiet_NaN();
  double backward = std::numeric_limits<double>::quiet_NaN();
  double disparity = std::numeric_limits<double>::quiet_NaN();
};

/// RDIC-inspired bounded risk + directional pixel uncertainty. Missing diagnostics
/// are neutral, not measured zero. This changes covariance, not track identities.
class TrackQuality {
public:
  static std::vector<Eigen::Matrix2d> covariance(const cv::Mat &previous, const cv::Mat &current,
                                               const std::vector<cv::Point2f> &old_points,
                                               const std::vector<cv::Point2f> &new_points,
                                               const std::vector<FlowQuality> &diagnostics,
                                               const std::vector<size_t> &ages);
  static double epipolar_error(const Eigen::Matrix3d &essential, const Eigen::Vector3d &left,
                               const Eigen::Vector3d &right, double focal);
};
} // namespace ov_core
#endif
