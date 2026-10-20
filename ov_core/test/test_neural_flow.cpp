/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <gtest/gtest.h>
#include <algorithm>
#include "cam/CamRadtan.h"
#include "track/NeuralFlow.h"
#include "track/TrackKLT.h"

using namespace ov_core;

#if ENABLE_KLTNET_ONNX
TEST(NeuralFlow, RealInferencePreservesReferenceAndMotionAcrossRekeyAndPrunesDeadTracks) {
  NeuralFlow tracker(SPARSE_TRACKER_FIXTURE);
  cv::Mat initial(80, 96, CV_8U, cv::Scalar(40)), later(80, 96, CV_8U, cv::Scalar(200));
  std::vector<cv::Point2f> points{{30, 30}, {60, 50}};
  auto first = tracker.predict(0, initial, later, points, {100, 101});
  ASSERT_EQ(first.points.size(), 2u);
  EXPECT_NEAR(first.points[0].x, 32, 1e-5);
  EXPECT_NEAR(first.points[0].y, 29, 1e-5);
  EXPECT_NEAR(first.confidence[0], .5 + 40. / 255, 1e-5);
  tracker.rekey(0, 100, 200);
  tracker.retain(0, {200});
  auto second = tracker.predict(0, later, later, {first.points[0]}, {200});
  EXPECT_NEAR(second.points[0].x, 35, 1e-5); // 2 px translation + 0.5 * previous 2 px flow.
  EXPECT_NEAR(second.points[0].y, 27.5, 1e-5);
  EXPECT_NEAR(second.confidence[0], first.confidence[0], 1e-5); // Reference is the first patch, not the previous patch.
  auto other_camera = tracker.predict(1, later, later, {points[0]}, {200});
  EXPECT_NEAR(other_camera.points[0].x, 32, 1e-5);
  EXPECT_NEAR(other_camera.confidence[0], .5 + 200. / 255, 1e-5);
  tracker.retain(0, {});
  auto new_track = tracker.predict(0, later, later, {points[0]}, {200});
  EXPECT_NEAR(new_track.points[0].x, 32, 1e-5);
  EXPECT_NEAR(new_track.confidence[0], other_camera.confidence[0], 1e-5);
  EXPECT_THROW(tracker.predict(0, later, later, points, {1}), std::invalid_argument);
}

class LearnedMatcher : public TrackKLT {
public:
  using TrackKLT::TrackKLT;
  using TrackKLT::perform_matching;
};

TEST(NeuralFlow, SelectedFrontendUsesPredictionsAndRejectsLowConfidence) {
  auto camera = std::make_shared<CamRadtan>(96, 80);
  Eigen::VectorXd calibration(8);
  calibration << 100, 100, 48, 40, 0, 0, 0, 0;
  camera->set_value(calibration);
  LearnedMatcher tracker({{0, camera}}, 100, 0, false, TrackBase::NONE, 10, 4, 4, 5);
  tracker.set_neural_tracker(SPARSE_TRACKER_FIXTURE, "cpu", .6);
  std::vector<cv::KeyPoint> previous, current;
  std::vector<size_t> ids;
  for (int y = 20; y <= 50; y += 10)
    for (int x = 20; x <= 60; x += 10) {
      previous.emplace_back(cv::Point2f(x, y), 1);
      ids.push_back(ids.size() + 1);
    }
  current = previous;
  cv::Mat image(80, 96, CV_8U, cv::Scalar(40));
  std::vector<uchar> mask;
  tracker.perform_matching({image}, {image}, previous, current, 0, 0, mask, nullptr, &ids);
  EXPECT_GE(std::count(mask.begin(), mask.end(), 1), 10);
  for (size_t i = 0; i < current.size(); ++i) {
    EXPECT_FLOAT_EQ(current[i].pt.x, previous[i].pt.x + 2);
    EXPECT_FLOAT_EQ(current[i].pt.y, previous[i].pt.y - 1);
  }
  tracker.set_neural_tracker(SPARSE_TRACKER_FIXTURE, "cpu", 2.0);
  tracker.perform_matching({image}, {image}, previous, current, 0, 0, mask, nullptr, &ids);
  EXPECT_EQ(std::count(mask.begin(), mask.end(), 1), 0);
}
#else
TEST(NeuralFlow, DisabledBuildRejectsInferenceSelection) {
  EXPECT_FALSE(NeuralFlow::available());
  EXPECT_THROW(NeuralFlow("missing-trained-model.onnx"), std::runtime_error);
}
#endif
