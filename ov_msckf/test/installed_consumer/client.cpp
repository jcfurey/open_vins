/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "cam/CamRadtan.h"
#include "core/VioManagerOptions.h"
#include "track/TrackAruco.h"
#include "track/TrackKLT.h"

#if !defined(ROS_AVAILABLE) || ROS_AVAILABLE != 2
#error The imported ROS 2 target must export ROS_AVAILABLE=2
#endif
#if !defined(ENABLE_ARUCO_TAGS)
#error The imported target must export the ArUco flag used by its public headers
#endif

int main() {
  cv::setNumThreads(1);
  auto camera = std::make_shared<ov_core::CamRadtan>(192, 128);
  Eigen::Matrix<double, 8, 1> intrinsics;
  intrinsics << 120, 120, 96, 64, 0, 0, 0, 0;
  camera->set_value(intrinsics);
  std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cameras{{0, camera}};
  ov_msckf::VioManagerOptions options;
  options.klt_pyramid_levels = 3;
  options.klt_window_size = 11;
  ov_core::TrackKLT klt(cameras, 50, 0, false, ov_core::TrackBase::HISTOGRAM, 20, 3, 2, 6,
                        options.klt_pyramid_levels, options.klt_window_size);
  ov_core::CameraData frame;
  frame.timestamp = 1.0;
  frame.sensor_ids = {0};
  frame.images = {cv::Mat::zeros(128, 192, CV_8UC1)};
  frame.masks = {cv::Mat::zeros(128, 192, CV_8UC1)};
  klt.feed_new_camera(frame);
#if ENABLE_ARUCO_TAGS
  // This public class has a conditional layout, so the flag is part of its ABI.
  ov_core::TrackAruco aruco(cameras, 32, false, ov_core::TrackBase::NONE, false);
  aruco.feed_new_camera(frame);
#endif
  return 0;
}
