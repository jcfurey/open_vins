/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <gtest/gtest.h>
#include <limits>

#include "ros/ROSVisualizerHelper.h"
#include "ros/Ros2Covariance.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/IMU.h"
#include "utils/quat_ops.h"

using namespace ov_msckf;

namespace {
ov_core::ImuData imu(double time, double value = 0) { return {time, Eigen::Vector3d::Constant(value), Eigen::Vector3d::Constant(value)}; }
std::shared_ptr<State> state() {
  StateOptions options;
  return std::make_shared<State>(options);
}
} // namespace

TEST(ImuInterpolation, IncludesExactEndpoints) {
  auto selected = Propagator::select_imu_readings({imu(0), imu(1, 1), imu(2, 2)}, 0, 2, false);
  ASSERT_EQ(selected.size(), 3U);
  EXPECT_DOUBLE_EQ(selected.front().timestamp, 0);
  EXPECT_DOUBLE_EQ(selected.back().timestamp, 2);
}

TEST(ImuInterpolation, CutsBothEndsOfOneInterval) {
  auto selected = Propagator::select_imu_readings({imu(0), imu(1, 10)}, 0.2, 0.8, false);
  ASSERT_EQ(selected.size(), 2U);
  EXPECT_DOUBLE_EQ(selected.front().timestamp, 0.2);
  EXPECT_DOUBLE_EQ(selected.back().timestamp, 0.8);
  EXPECT_NEAR(selected.front().am(0), 2, 1e-12);
  EXPECT_NEAR(selected.back().am(0), 8, 1e-12);
}

TEST(ImuInterpolation, RefusesUnbracketedAndEmptyIntervals) {
  std::vector<ov_core::ImuData> samples = {imu(1), imu(2)};
  EXPECT_TRUE(Propagator::select_imu_readings(samples, 0.5, 1.5, false).empty());
  EXPECT_TRUE(Propagator::select_imu_readings(samples, 1.5, 2.5, false).empty());
  EXPECT_TRUE(Propagator::select_imu_readings(samples, 1.5, 1.5, false).empty());
  EXPECT_TRUE(Propagator::select_imu_readings({imu(1)}, 1, 2, false).empty());
}

TEST(ImuInterpolation, RejectsANonfiniteFirstEndpoint) {
  auto bad = imu(1);
  bad.am(0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(Propagator::select_imu_readings({bad, imu(2)}, 1, 2, false).empty());
}

TEST(Propagation, MissingImuDoesNotAdvanceStateOrClone) {
  auto estimate = state();
  estimate->_timestamp = 1;
  const auto covariance = StateHelper::get_full_covariance(estimate);
  NoiseManager noises;
  Propagator propagator(noises, 9.81);
  propagator.feed_imu(imu(1.2));
  propagator.feed_imu(imu(1.4));
  propagator.propagate_and_clone(estimate, 1.5);
  EXPECT_DOUBLE_EQ(estimate->_timestamp, 1);
  EXPECT_TRUE(estimate->_clones_IMU.empty());
  EXPECT_EQ((StateHelper::get_full_covariance(estimate) - covariance).norm(), 0);
}

TEST(Propagation, FastPredictionDoesNotApplyTimeOffsetTwice) {
  auto estimate = state();
  estimate->_timestamp = 10;
  Eigen::Matrix<double, 1, 1> offset;
  offset << 0.02;
  estimate->_calib_dt_CAMtoIMU->set_value(offset);
  auto value = estimate->_imu->value();
  value(7) = 1;
  estimate->_imu->set_value(value);
  NoiseManager noises;
  Propagator propagator(noises, 9.81);
  for (double time : {10.015, 10.025, 10.035}) {
    auto sample = imu(time);
    sample.am << 0, 0, 9.81;
    propagator.feed_imu(sample);
  }
  Eigen::Matrix<double, 13, 1> predicted;
  Eigen::Matrix<double, 12, 12> covariance;
  ASSERT_TRUE(propagator.fast_state_propagate(estimate, 10.03, predicted, covariance));
  EXPECT_NEAR(predicted(4), 0.01, 1e-12);
  EXPECT_NEAR(predicted(7), 1, 1e-12);
  EXPECT_TRUE(covariance.allFinite());
}

TEST(Ros2Covariance, RotatesLocalAnglesAndCrossTermsIntoParentFrame) {
  Eigen::Matrix<double, 6, 6> local = Eigen::Matrix<double, 6, 6>::Zero();
  local.diagonal() << 1, 2, 3, 4, 5, 6;
  local(0, 3) = local(3, 0) = 0.5;
  const Eigen::Matrix3d R_ItoG = ov_core::exp_so3(Eigen::Vector3d(0, 0, M_PI_2));
  const auto ros = ros_pose_covariance(local, R_ItoG.transpose());
  EXPECT_NEAR(ros(3, 3), 5, 1e-12);
  EXPECT_NEAR(ros(4, 4), 4, 1e-12);
  EXPECT_NEAR(ros(0, 4), 0.5, 1e-12);
  EXPECT_NEAR(ros(4, 0), 0.5, 1e-12);
  EXPECT_NEAR((ros - ros.transpose()).norm(), 0, 1e-12);
}

TEST(Propagation, BodyTwistIncludesOrientationAndGyroBiasUncertainty) {
  auto estimate = state();
  estimate->_timestamp = 1;
  auto value = estimate->_imu->value();
  value(7) = 1;
  estimate->_imu->set_value(value);
  Eigen::Matrix<double, 15, 15> initial = Eigen::Matrix<double, 15, 15>::Zero();
  initial.block<3, 3>(0, 0) = 0.2 * Eigen::Matrix3d::Identity();
  initial.block<3, 3>(9, 9).setIdentity();
  StateHelper::set_initial_covariance(estimate, initial, {estimate->_imu});
  NoiseManager noises;
  noises.sigma_w = noises.sigma_wb = noises.sigma_a = noises.sigma_ab = 0;
  Propagator propagator(noises, 0);
  propagator.feed_imu(imu(1));
  propagator.feed_imu(imu(1.01));
  Eigen::Matrix<double, 13, 1> predicted;
  Eigen::Matrix<double, 12, 12> covariance;
  ASSERT_TRUE(propagator.fast_state_propagate(estimate, 1.01, predicted, covariance));
  EXPECT_NEAR(covariance(7, 7), 0.2001, 1e-12);
  EXPECT_NEAR(covariance(11, 11), 1, 1e-12);
  EXPECT_NEAR(covariance(7, 11), -0.01, 1e-12);
  EXPECT_NEAR((covariance - covariance.transpose()).norm(), 0, 1e-12);
}

TEST(EkfUpdate, RejectsSingularInnovationWithoutChangingState) {
  auto estimate = state();
  const auto covariance = StateHelper::get_full_covariance(estimate);
  const auto before = estimate->_imu->value();
  const Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 15);
  StateHelper::EKFUpdate(estimate, {estimate->_imu}, H, Eigen::Vector3d::Ones(), Eigen::Matrix3d::Zero());
  EXPECT_EQ((StateHelper::get_full_covariance(estimate) - covariance).norm(), 0);
  EXPECT_EQ((estimate->_imu->value() - before).norm(), 0);
}

TEST(EkfUpdate, MatchesKnownLinearPosterior) {
  auto estimate = state();
  StateHelper::set_initial_covariance(estimate, 0.2 * Eigen::Matrix<double, 15, 15>::Identity(), {estimate->_imu});
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 15);
  H.block<3, 3>(0, 3).setIdentity();
  StateHelper::EKFUpdate(estimate, {estimate->_imu}, H, Eigen::Vector3d(1, 2, 3), 0.3 * Eigen::Matrix3d::Identity());
  EXPECT_NEAR((estimate->_imu->pos() - Eigen::Vector3d(0.4, 0.8, 1.2)).norm(), 0, 1e-12);
  const auto posterior = StateHelper::get_marginal_covariance(estimate, {estimate->_imu});
  EXPECT_NEAR((posterior.block<3, 3>(3, 3) - 0.12 * Eigen::Matrix3d::Identity()).norm(), 0, 1e-12);
}

TEST(Ros2PointCloud, EmptyFeaturesProduceAValidEmptyCloud) {
  const auto cloud = ROSVisualizerHelper::get_ros_pointcloud(nullptr, {});
  EXPECT_EQ(cloud.width, 0U);
  EXPECT_EQ(cloud.height, 1U);
  EXPECT_EQ(cloud.fields.size(), 3U);
  EXPECT_GT(cloud.point_step, 0U);
  EXPECT_EQ(cloud.row_step, 0U);
  EXPECT_TRUE(cloud.data.empty());
}

TEST(Ros2PointCloud, SerializesCoordinatesInDeclaredFields) {
  const auto cloud = ROSVisualizerHelper::get_ros_pointcloud(nullptr, {Eigen::Vector3d(1, 2, 3), Eigen::Vector3d(-4, -5, -6)});
  ASSERT_EQ(cloud.width, 2U);
  EXPECT_EQ(cloud.data.size(), cloud.row_step);
  sensor_msgs::PointCloud2ConstIterator<float> x(cloud, "x"), y(cloud, "y"), z(cloud, "z");
  EXPECT_FLOAT_EQ(*x, 1);
  EXPECT_FLOAT_EQ(*y, 2);
  EXPECT_FLOAT_EQ(*z, 3);
  ++x;
  ++y;
  ++z;
  EXPECT_FLOAT_EQ(*x, -4);
  EXPECT_FLOAT_EQ(*y, -5);
  EXPECT_FLOAT_EQ(*z, -6);
}
