/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <numeric>

#include "cam/CamRadtan.h"
#include "track/TrackKLT.h"

namespace {
class Tracker : public ov_core::TrackKLT {
public:
  using TrackKLT::geometric_inlier_mask;
  using TrackKLT::perform_matching;
  using TrackKLT::TrackKLT;
};

std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cameras() {
  auto camera = std::make_shared<ov_core::CamRadtan>(320, 240);
  Eigen::Matrix<double, 8, 1> intrinsics;
  intrinsics << 200, 200, 160, 120, 0, 0, 0, 0;
  camera->set_value(intrinsics);
  return {{0, camera}};
}

void project_scene(std::vector<cv::Point2f> &a, std::vector<cv::Point2f> &b, int count, int seed, double baseline) {
  cv::RNG random(seed);
  for (int i = 0; i < count; ++i) {
    const double x = random.uniform(-1.5, 1.5), y = random.uniform(-1.0, 1.0), z = random.uniform(3.0, 9.0);
    a.emplace_back(200 * x / z + 160, 200 * y / z + 120);
    // Non-planar points with a small rotation and horizontal translation.
    const double yaw = baseline > 0 ? .05 : -.15;
    const double xx = std::cos(yaw) * x + std::sin(yaw) * z - baseline;
    const double zz = -std::sin(yaw) * x + std::cos(yaw) * z;
    b.emplace_back(200 * xx / zz + 160, 200 * (y + (baseline > 0 ? .02 : -.6)) / zz + 120);
  }
}
} // namespace

TEST(KltOutliers, FailedPredictionsCannotChangeTheGeometricModel) {
  Tracker tracker(cameras(), 300, 0, false, ov_core::TrackBase::NONE, 10, 4, 3, 6);
  std::vector<cv::Point2f> a, b;
  project_scene(a, b, 60, 77, .4);
  const auto reference = tracker.geometric_inlier_mask(a, b, std::vector<uchar>(a.size(), 1), 0, 0);
  ASSERT_EQ(std::accumulate(reference.begin(), reference.end(), 0), 60);

  // Failed LK predictions form a much larger, incompatible consensus. They must never reach RANSAC.
  project_scene(a, b, 600, 81, -.8);
  std::vector<uchar> status(a.size(), 0);
  std::fill(status.begin(), status.begin() + 60, 1);
  const auto legacy = tracker.geometric_inlier_mask(a, b, status, 0, 0, false);
  EXPECT_LT(std::accumulate(legacy.begin(), legacy.begin() + 60, 0), 30);
  // A successful LK status alone also does not make NaN/Inf/out-of-image predictions usable.
  for (const auto bad :
       {cv::Point2f(std::numeric_limits<float>::quiet_NaN(), 100), cv::Point2f(100, std::numeric_limits<float>::infinity()),
        cv::Point2f(-1, 100), cv::Point2f(320, 100), cv::Point2f(100, 240)}) {
    a.emplace_back(100, 100);
    b.push_back(bad);
    status.push_back(1);
  }
  const auto actual = tracker.geometric_inlier_mask(a, b, status, 0, 0);
  ASSERT_EQ(actual.size(), a.size());
  EXPECT_TRUE(std::equal(reference.begin(), reference.end(), actual.begin()));
  EXPECT_EQ(std::accumulate(actual.begin() + 60, actual.end(), 0), 0);
}

TEST(KltOutliers, MinimumCorrespondenceCountUsesSuccessfulTracks) {
  Tracker tracker(cameras(), 300, 0, false, ov_core::TrackBase::NONE, 10, 4, 3, 6);
  std::vector<cv::Point2f> a, b;
  project_scene(a, b, 60, 77, .4);
  std::vector<uchar> status(a.size(), 0);
  std::fill(status.begin(), status.begin() + 9, 1);
  const auto actual = tracker.geometric_inlier_mask(a, b, status, 0, 0);
  EXPECT_EQ(actual.size(), a.size());
  EXPECT_EQ(std::accumulate(actual.begin(), actual.end(), 0), 0);
}

TEST(KltOutliers, RoundTripCheckRejectsNonreciprocalMatchesAndPreservesTranslation) {
  cv::setNumThreads(1);
  cv::Mat source(240, 320, CV_8UC1), target;
  cv::RNG random(42);
  random.fill(source, cv::RNG::UNIFORM, 0, 256);
  cv::GaussianBlur(source, source, cv::Size(3, 3), .8);
  const cv::Mat affine = (cv::Mat_<double>(2, 3) << 1, 0, 3.3, 0, 1, -1.7);
  cv::warpAffine(source, target, affine, source.size());
  // Occlude part of the frame with unrelated textured content.
  source(cv::Rect(10, 10, 100, 100)).copyTo(target(cv::Rect(110, 70, 100, 100)));
  std::vector<cv::Point2f> points;
  cv::goodFeaturesToTrack(source, points, 350, .01, 8);
  std::vector<cv::KeyPoint> old, forward;
  for (const auto &point : points)
    old.emplace_back(point, 7);
  forward = old;
  std::vector<cv::Mat> source_pyramid, target_pyramid;
  cv::buildOpticalFlowPyramid(source, source_pyramid, cv::Size(15, 15), 3);
  cv::buildOpticalFlowPyramid(target, target_pyramid, cv::Size(15, 15), 3);
  Tracker one_way(cameras(), 350, 0, false, ov_core::TrackBase::NONE, 10, 4, 3, 6, 3, 15, 0);
  Tracker reciprocal(cameras(), 350, 0, false, ov_core::TrackBase::NONE, 10, 4, 3, 6, 3, 15, .5);
  std::vector<uchar> one_way_status, reciprocal_status;
  auto old_copy = old, forward_copy = forward;
  one_way.perform_matching(source_pyramid, target_pyramid, old, forward, 0, 0, one_way_status);
  reciprocal.perform_matching(source_pyramid, target_pyramid, old_copy, forward_copy, 0, 0, reciprocal_status);
  ASSERT_EQ(reciprocal_status.size(), points.size());
  EXPECT_GT(std::accumulate(reciprocal_status.begin(), reciprocal_status.end(), 0), 100);

  std::vector<cv::Point2f> back, target_points;
  std::vector<size_t> indices;
  for (size_t i = 0; i < points.size(); ++i) {
    if (!one_way_status[i] && !reciprocal_status[i])
      continue;
    back.push_back(points[i]);
    target_points.push_back(forward_copy[i].pt);
    indices.push_back(i);
  }
  std::vector<uchar> reverse_status;
  std::vector<float> error;
  cv::calcOpticalFlowPyrLK(target_pyramid, source_pyramid, target_points, back, reverse_status, error, cv::Size(15, 15), 3,
                           cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, .01), cv::OPTFLOW_USE_INITIAL_FLOW);
  size_t rejected_nonreciprocal = 0, accurate = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    const size_t index = indices[i];
    const bool consistent = reverse_status[i] && cv::norm(back[i] - points[index]) <= .5;
    if (reciprocal_status[index]) {
      EXPECT_TRUE(consistent);
      accurate += cv::norm(target_points[i] - points[index] - cv::Point2f(3.3, -1.7)) < .2;
    }
    rejected_nonreciprocal += one_way_status[index] && !consistent && !reciprocal_status[index];
  }
  EXPECT_GT(rejected_nonreciprocal, 0u);
  EXPECT_GT(accurate, 100u);
}

TEST(KltOutliers, InvalidRoundTripThresholdIsRejected) {
  for (double limit : {-1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    EXPECT_THROW((Tracker(cameras(), 100, 0, false, ov_core::TrackBase::NONE, 10, 4, 3, 6, 3, 15, limit)), std::invalid_argument);
  }
}
