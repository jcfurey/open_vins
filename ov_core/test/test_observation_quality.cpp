/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <gtest/gtest.h>
#include <Eigen/Eigenvalues>
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "track/TrackQuality.h"
#include "utils/quat_ops.h"

using namespace ov_core;

TEST(ObservationCovariance, MetadataSurvivesCloneAppendAndEveryPruningPolicy) {
  auto database = std::make_shared<FeatureDatabase>();
  const Eigen::Matrix2d first = (Eigen::Matrix2d() << 3, .2, .2, 2).finished();
  const Eigen::Matrix2d second = 4.0 * Eigen::Matrix2d::Identity();
  for (size_t camera : {0u, 1u}) {
    for (int time = 1; time <= 4; ++time)
      database->update_feature(42, time, camera, 20, 30, .1, .2, time == 2 ? nullptr : &first);
  }
  Feature snapshot;
  ASSERT_TRUE(database->get_feature_clone(42, snapshot));
  EXPECT_EQ(snapshot.observation_covariances.at(1).size(), 3u);
  auto appended = std::make_shared<FeatureDatabase>();
  appended->append_new_measurements(database);
  database->update_feature(42, 5, 0, 20, 30, .1, .2, &second);
  appended->append_new_measurements(database);
  auto feature = appended->get_feature(42);
  EXPECT_TRUE(feature->observation_covariances.at(0).at(5).isApprox(second));
  feature->clean_invalid_measurements({1});
  EXPECT_EQ(feature->observation_covariances.at(1).count(1), 0u);
  feature->clean_older_measurements(2);
  EXPECT_EQ(feature->observation_covariances.at(0).count(2), 0u);
  feature->clean_old_measurements({4, 5});
  ASSERT_EQ(feature->timestamps.at(0), (std::vector<double>{4, 5}));
  ASSERT_EQ(feature->observation_covariances.at(0).size(), 2u);
  EXPECT_TRUE(feature->observation_covariances.at(0).at(4).isApprox(first));
  EXPECT_EQ(snapshot.observation_covariances.at(0).size(), 3u); // Independent clone, not shared metadata storage.
}

TEST(ObservationCovariance, OriginalTrackerObservationsAllocateNoMetadata) {
  FeatureDatabase database;
  database.update_feature(1, 1, 0, 10, 10, .1, .1);
  EXPECT_TRUE(database.get_feature(1)->observation_covariances.empty());
}

TEST(TrackingQuality, EpipolarErrorUsesTheCalibratedStereoGeometry) {
  Eigen::Matrix3d rotation = Eigen::AngleAxisd(.03, Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Eigen::Vector3d translation(-.1, .01, .002), point(.2, -.1, 3.0);
  const Eigen::Matrix3d essential = skew_x(translation) * rotation;
  const Eigen::Vector3d target = rotation * point + translation;
  const Eigen::Vector3d left = point / point.z(), right = target / target.z();
  EXPECT_NEAR(TrackQuality::epipolar_error(essential, left, right, 450), 0, 1e-12);
  Eigen::Vector3d wrong = right;
  wrong.y() += .02;
  EXPECT_GT(TrackQuality::epipolar_error(essential, left, wrong, 450), 5.0);
  EXPECT_TRUE(std::isnan(TrackQuality::epipolar_error(Eigen::Matrix3d::Zero(), left, right, 450)));
}

TEST(TrackingQuality, DirectionalPenaltyFollowsTheApertureDirection) {
  cv::Mat image(100, 100, CV_8UC1);
  for (int y = 0; y < image.rows; ++y)
    for (int x = 0; x < image.cols; ++x)
      image.at<uchar>(y, x) = 2 * x;
  const std::vector<cv::Point2f> points{{50, 50}, {30, 30}};
  const auto covariance = TrackQuality::covariance(image, image, points, points, std::vector<FlowQuality>(2), {5, 5});
  EXPECT_GT(covariance[0](1, 1), covariance[0](0, 0) * 2);
  EXPECT_NEAR(covariance[0](0, 1), 0, 1e-10);
  cv::Mat transposed;
  cv::transpose(image, transposed);
  const auto rotated = TrackQuality::covariance(transposed, transposed, points, points, std::vector<FlowQuality>(2), {5, 5});
  EXPECT_NEAR(covariance[0](1, 1), rotated[0](0, 0), 1e-10);
  EXPECT_NEAR(covariance[0](0, 0), rotated[0](1, 1), 1e-10);
}

TEST(TrackingQuality, SuspiciousTracksArePenalizedWithBoundedPositiveCovariance) {
  cv::Mat image(100, 100, CV_8UC1);
  cv::RNG random(7);
  random.fill(image, cv::RNG::UNIFORM, 0, 256);
  std::vector<cv::Point2f> points(9, cv::Point2f(50, 50));
  std::vector<FlowQuality> diagnostics(9);
  diagnostics.back().temporal = 100;
  diagnostics.back().backward = 10;
  auto covariance = TrackQuality::covariance(image, image, points, points, diagnostics, std::vector<size_t>(9, 5));
  EXPECT_GT(covariance.back().trace(), 4 * covariance.front().trace());
  for (const auto &value : covariance) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen(value);
    EXPECT_GE(eigen.eigenvalues().minCoeff(), 1.0 - 1e-10);
    EXPECT_LE(eigen.eigenvalues().maxCoeff(), 49.0 + 1e-10);
  }
}
