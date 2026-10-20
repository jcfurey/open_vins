/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "cam/CamRadtan.h"
#include "track/LearnedFeatures.h"
#include "track/TrackKLT.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace ov_core;

TEST(LearnedFeatures, LocalPeaksRespectExclusionMasksGridBudgetsAndImageBorders) {
  cv::Mat score = cv::Mat::zeros(64, 96, CV_32F), mask = cv::Mat::zeros(64, 96, CV_8U);
  score.at<float>(15, 15) = .8;
  score.at<float>(20, 25) = .9;
  score.at<float>(45, 75) = .99;
  mask.at<uchar>(45, 75) = 255;
  score.at<float>(40, 65) = .7;
  score.at<float>(2, 2) = 1;
  score.at<float>(15, 65) = .95; // A cell not requested for replenishment.
  std::vector<cv::KeyPoint> points;
  LearnedFeatures::detect(score, mask, {{0, 0}, {1, 1}}, points, 1, 2, 2, .5);
  ASSERT_EQ(points.size(), 2u);
  EXPECT_EQ(points[0].pt, cv::Point2f(25, 20));
  EXPECT_EQ(points[1].pt, cv::Point2f(65, 40));
  EXPECT_THROW(LearnedFeatures::detect(score, cv::Mat(), {{0, 0}}, points, 1, 2, 2, .5), std::invalid_argument);
}

#if ENABLE_LETNET_ONNX && defined(LETNET_TEST_MODEL)
TEST(LearnedFeatures, AuthorTrainedInferenceKeepsOriginalResolutionAndOwnsPriorScores) {
  cv::setNumThreads(2);
  LearnedFeatures network(LETNET_TEST_MODEL);
  cv::Mat image(480, 752, CV_8U);
  cv::RNG random(41);
  random.fill(image, cv::RNG::UNIFORM, 0, 256);
  auto first = network.infer(image);
  const cv::Mat saved = first.score.clone();
  ASSERT_EQ(first.tracking.size(), image.size());
  ASSERT_EQ(first.tracking.type(), CV_8UC3);
  ASSERT_EQ(first.score.type(), CV_32FC1);
  EXPECT_TRUE(cv::checkRange(first.score, true, nullptr, 0, 1.000001));
  auto second = network.infer(cv::Mat::zeros(128, 192, CV_8U)); // Also exercise a different spatial shape.
  EXPECT_EQ(second.score.size(), cv::Size(192, 128));
  EXPECT_EQ(cv::norm(first.score, saved, cv::NORM_INF), 0);
  EXPECT_THROW(network.infer(cv::Mat()), std::invalid_argument);
  EXPECT_THROW(network.infer(first.tracking), std::invalid_argument);
}

namespace {
class Tracker : public TrackKLT {
public:
  using TrackKLT::TrackKLT;
  cv::Mat raw(size_t camera) const { return img_last.at(camera); }
  int tracking_channels(size_t camera) const { return img_pyramid_last.at(camera).at(0).channels(); }
};
} // namespace

TEST(LearnedFeatures, StereoTracksKnownPixelMotionUsingLearnedFeaturesWhileRetainingRawViews) {
  cv::setNumThreads(2);
  std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras;
  for (size_t id = 0; id < 2; ++id) {
    auto camera = std::make_shared<CamRadtan>(320, 240);
    Eigen::Matrix<double, 8, 1> k;
    k << 220, 220, 160, 120, 0, 0, 0, 0;
    camera->set_value(k);
    cameras[id] = camera;
  }
  Tracker tracker(cameras, 150, 0, true, TrackBase::NONE, 10, 4, 4, 6, 3, 15);
  tracker.set_letnet_tracker(LETNET_TEST_MODEL, .2);
  cv::Mat texture(280, 380, CV_8U);
  cv::RNG random(22);
  random.fill(texture, cv::RNG::UNIFORM, 0, 256);
  cv::GaussianBlur(texture, texture, cv::Size(3, 3), .6);
  std::unordered_map<size_t, std::vector<size_t>> previous_ids;
  std::unordered_map<size_t, std::vector<cv::KeyPoint>> previous_points;
  size_t matched = 0;
  double squared_error = 0;
  for (int frame = 0; frame < 6; ++frame) {
    CameraData data;
    data.timestamp = 1 + .05 * frame;
    for (size_t camera = 0; camera < 2; ++camera) {
      data.sensor_ids.push_back(camera);
      data.images.push_back(texture(cv::Rect(10 + 2 * frame + 4 * camera, 10 + frame, 320, 240)).clone());
      data.masks.push_back(cv::Mat::zeros(240, 320, CV_8U));
    }
    tracker.feed_new_camera(data);
    const auto ids = tracker.get_last_ids();
    const auto points = tracker.get_last_obs();
    for (size_t camera = 0; camera < 2; ++camera) {
      EXPECT_EQ(cv::norm(tracker.raw(camera), data.images[camera], cv::NORM_INF), 0);
      EXPECT_EQ(tracker.tracking_channels(camera), 3);
      if (frame == 0)
        continue;
      for (size_t i = 0; i < ids.at(camera).size(); ++i) {
        auto found = std::find(previous_ids[camera].begin(), previous_ids[camera].end(), ids.at(camera)[i]);
        if (found == previous_ids[camera].end())
          continue;
        const size_t old = found - previous_ids[camera].begin();
        const auto error = points.at(camera)[i].pt - previous_points[camera][old].pt - cv::Point2f(-2, -1);
        squared_error += error.dot(error);
        ++matched;
      }
    }
    previous_ids = ids;
    previous_points = points;
  }
  ASSERT_GT(matched, 100u);
  EXPECT_LT(std::sqrt(squared_error / matched), .3);
}
#elif !ENABLE_LETNET_ONNX
TEST(LearnedFeatures, DisabledBuildRejectsModelSelectionExplicitly) {
  EXPECT_FALSE(LearnedFeatures::available());
  EXPECT_THROW(LearnedFeatures("missing-model.onnx"), std::runtime_error);
}
#endif
