/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <gtest/gtest.h>

#include <algorithm>

#include "cam/CamRadtan.h"
#include "track/TrackDescriptor.h"
#include "track/TrackKLT.h"

namespace {

constexpr int num_cameras = 3;

std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cameras() {
  std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> result;
  for (size_t i = 0; i < num_cameras; ++i) {
    auto model = std::make_shared<ov_core::CamRadtan>(160, 120);
    Eigen::Matrix<double, 8, 1> intrinsics;
    intrinsics << 100, 100, 80, 60, 0, 0, 0, 0;
    model->set_value(intrinsics);
    result[i] = model;
  }
  return result;
}

ov_core::CameraData textured_frame(double time) {
  ov_core::CameraData frame;
  frame.timestamp = time;
  cv::RNG random(7);
  for (int i = 0; i < num_cameras; ++i) {
    cv::Mat image(120, 160, CV_8UC1);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    frame.sensor_ids.push_back(i);
    frame.images.push_back(image);
    frame.masks.push_back(cv::Mat::zeros(120, 160, CV_8UC1));
  }
  return frame;
}

/// New feature ids must form consecutive per-camera blocks in camera order, whatever the thread interleaving.
void expect_ids_in_camera_order(ov_core::TrackBase &tracker) {
  tracker.feed_new_camera(textured_frame(1.0));
  const auto ids = tracker.get_last_ids();
  size_t expected = 2; // the first id handed out when there are no aruco tags
  for (size_t camera = 0; camera < num_cameras; ++camera) {
    auto sorted = ids.at(camera);
    ASSERT_FALSE(sorted.empty());
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(sorted.front(), expected) << "camera " << camera;
    expected = sorted.back() + 1;
  }
}

} // namespace

TEST(BinocularTracking, KltAssignsIdsInCameraOrder) {
  cv::setNumThreads(4);
  for (int repeat = 0; repeat < 5; ++repeat) {
    ov_core::TrackKLT tracker(cameras(), 200, 0, false, ov_core::TrackBase::NONE, 10, 4, 3, 5);
    expect_ids_in_camera_order(tracker);
  }
}

TEST(BinocularTracking, DescriptorAssignsIdsInCameraOrder) {
  cv::setNumThreads(4);
  for (int repeat = 0; repeat < 5; ++repeat) {
    ov_core::TrackDescriptor tracker(cameras(), 200, 0, false, ov_core::TrackBase::NONE, 10, 4, 3, 5, 0.7);
    expect_ids_in_camera_order(tracker);
  }
}
