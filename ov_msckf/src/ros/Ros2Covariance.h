/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef OV_MSCKF_ROS2_COVARIANCE_H
#define OV_MSCKF_ROS2_COVARIANCE_H

#include <Eigen/Core>

namespace ov_msckf {

/// ROS fixed-axis orientation errors are expressed in the pose's parent frame.
inline Eigen::Matrix<double, 6, 6> ros_pose_covariance(const Eigen::Matrix<double, 6, 6> &position_local_angle_covariance,
                                                       const Eigen::Matrix3d &R_GtoI) {
  Eigen::Matrix<double, 6, 6> projection = Eigen::Matrix<double, 6, 6>::Identity();
  projection.block<3, 3>(3, 3) = R_GtoI.transpose();
  return projection * position_local_angle_covariance * projection.transpose();
}

} // namespace ov_msckf
#endif
