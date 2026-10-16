/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <gtest/gtest.h>

#include "cam/CamRadtan.h"
#include "track/TrackKLT.h"

class StereoDetector : public ov_core::TrackKLT {
public:
  using TrackKLT::perform_detection_stereo;
  using TrackKLT::TrackKLT;
};

TEST(StereoMask, RightDetectorUsesTheRightMask) {
  std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cameras;
  for (size_t i = 0; i < 2; ++i) {
    auto model = std::make_shared<ov_core::CamRadtan>(192, 128);
    Eigen::Matrix<double, 8, 1> intrinsics;
    intrinsics << 120, 120, 96, 64, 0, 0, 0, 0;
    model->set_value(intrinsics);
    cameras[i] = model;
  }
  StereoDetector detector(cameras, 100, 0, true, ov_core::TrackBase::HistogramMethod::NONE, 10, 3, 2, 8);
  cv::Mat left = cv::Mat::zeros(128, 192, CV_8UC1);
  cv::Mat right = left.clone();
  cv::Mat left_mask = left.clone(), right_mask = left.clone();
  cv::RNG random(42);
  auto texture = right(cv::Rect(88, 20, 16, 88));
  random.fill(texture, cv::RNG::UNIFORM, 0, 256);
  right_mask(cv::Rect(80, 0, 32, 128)).setTo(255);
  std::vector<cv::KeyPoint> left_points, right_points;
  std::vector<size_t> left_ids, right_ids;
  detector.perform_detection_stereo({left}, {right}, left_mask, right_mask, 0, 1, left_points, right_points, left_ids, right_ids);
  EXPECT_TRUE(left_points.empty());
  EXPECT_TRUE(right_points.empty()) << "The only right-image texture is inside its exclusion mask";
}
