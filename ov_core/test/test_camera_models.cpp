/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <gtest/gtest.h>

#include <random>

#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"

namespace {

/// The batched call must reproduce the per-point result exactly, so it cannot change what the estimator sees.
void expect_batch_matches_single_points(ov_core::CamBase &camera) {
  std::mt19937 generator(5);
  std::uniform_real_distribution<float> u(0, camera.w() - 1), v(0, camera.h() - 1);
  std::vector<cv::Point2f> pixels(500);
  for (auto &pixel : pixels)
    pixel = {u(generator), v(generator)};

  const std::vector<cv::Point2f> batch = camera.undistort_cv(pixels);
  ASSERT_EQ(batch.size(), pixels.size());
  for (size_t i = 0; i < pixels.size(); i++) {
    const cv::Point2f single = camera.undistort_cv(pixels.at(i));
    EXPECT_EQ(batch.at(i).x, single.x) << "point " << i;
    EXPECT_EQ(batch.at(i).y, single.y) << "point " << i;
  }
  EXPECT_TRUE(camera.undistort_cv(std::vector<cv::Point2f>()).empty());
}

} // namespace

TEST(CameraModels, BatchedRadtanUndistortionMatchesSinglePoints) {
  ov_core::CamRadtan camera(752, 480);
  Eigen::Matrix<double, 8, 1> intrinsics;
  intrinsics << 458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05;
  camera.set_value(intrinsics);
  expect_batch_matches_single_points(camera);
}

TEST(CameraModels, BatchedEquidistantUndistortionMatchesSinglePoints) {
  ov_core::CamEqui camera(512, 512);
  Eigen::Matrix<double, 8, 1> intrinsics;
  intrinsics << 190.978, 190.973, 254.932, 256.897, 0.00348239, 0.000715035, -0.00205324, 0.000202937;
  camera.set_value(intrinsics);
  expect_batch_matches_single_points(camera);
}
