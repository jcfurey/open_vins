/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

#include "ros/ROSVisualizerHelper.h"
#include "core/VioManager.h"
#include "cam/CamRadtan.h"
#include "track/TrackSIM.h"
#include "ros/Ros2Covariance.h"
#include "ros/Ros2ResetOptions.h"
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

namespace {
VioManagerOptions stereo_options() {
  VioManagerOptions options;
  options.state_options.num_cameras = 2;
  options.init_options.num_cameras = 2;
  options.use_aruco = false;
  options.vec_dw << 1, 0, 0, 1, 0, 1;
  options.vec_da = options.vec_dw;
  options.vec_tg.setZero();
  options.q_ACCtoIMU << 0, 0, 0, 1;
  options.q_GYROtoIMU = options.q_ACCtoIMU;
  options.featinit_options.max_cond_number = 1e6;
  for (size_t id = 0; id < 2; ++id) {
    auto camera = std::make_shared<ov_core::CamRadtan>(320, 240);
    Eigen::Matrix<double, 8, 1> calibration;
    calibration << 200, 200, 160, 120, 0, 0, 0, 0;
    camera->set_value(calibration);
    options.camera_intrinsics[id] = camera;
    Eigen::VectorXd extrinsic(7);
    extrinsic << 0, 0, 0, 1, -static_cast<double>(id), 0, 0;
    options.camera_extrinsics[id] = extrinsic;
  }
  options.init_options.camera_intrinsics = options.camera_intrinsics;
  options.init_options.camera_extrinsics = options.camera_extrinsics;
  return options;
}

class ActiveTrackManager : public VioManager {
public:
  explicit ActiveTrackManager(VioManagerOptions &options) : VioManager(options) {
    tracker = std::make_shared<ov_core::TrackSIM>(options.camera_intrinsics, 0);
    trackFEATS = tracker;
    Eigen::Matrix<double, 17, 1> truth = Eigen::Matrix<double, 17, 1>::Zero();
    truth(0) = 10;
    truth(4) = 1;
    initialize_with_gt(truth);
  }
  void frame(double time) {
    state->_timestamp = time;
    StateHelper::augment_clone(state, Eigen::Vector3d::Zero());
    Eigen::VectorXf left(2), right(2);
    left << 160, 120;
    right << 120, 120;
    tracker->feed_measurement_simulation(time, {0, 1}, {{{42, left}}, {{42, right}}});
    ov_core::CameraData message;
    message.timestamp = time;
    message.sensor_ids = {0, 1};
    message.images = {cv::Mat::zeros(240, 320, CV_8UC1), cv::Mat::zeros(240, 320, CV_8UC1)};
    retriangulate_active_tracks(message);
  }
  const auto &points() const { return active_tracks_posinG; }
  const auto &counts() const { return active_feat_linsys_count; }
private:
  std::shared_ptr<ov_core::TrackSIM> tracker;
};
} // namespace

TEST(ActiveStereoTracks, AccumulatesBothBearingsWithoutTemporalMotion) {
  auto options = stereo_options();
  ActiveTrackManager manager(options);
  manager.set_active_image_enabled(false);
  manager.frame(10);
  ASSERT_EQ(manager.counts().size(), 1U);
  EXPECT_EQ(manager.counts().begin()->second, 2);
  manager.frame(10.05);
  ASSERT_EQ(manager.points().size(), 1U);
  EXPECT_NEAR((manager.points().begin()->second - Eigen::Vector3d(0, 0, 5)).norm(), 0, 1e-6);
  double time;
  cv::Mat image;
  manager.get_active_image(time, image);
  EXPECT_TRUE(image.empty());
  // A consumer arriving later gets an image while keeping accumulated geometry.
  manager.set_active_image_enabled(true);
  manager.frame(10.1);
  manager.get_active_image(time, image);
  EXPECT_EQ(image.size(), cv::Size(320, 240));
  EXPECT_EQ(manager.counts().begin()->second, 6);
}

TEST(Ros2ResetCalibration, RestoresIndependentCameraModelsAcrossRepeatedEpochs) {
  auto original = stereo_options();
  auto fisheye = std::make_shared<ov_core::CamEqui>(320, 240);
  fisheye->set_value(original.camera_intrinsics.at(1)->get_value());
  original.camera_intrinsics[1] = original.init_options.camera_intrinsics[1] = fisheye;
  const auto snapshot = copy_ros2_reset_options(original);
  auto changed = original.camera_intrinsics.at(0)->get_value();
  changed(0) = 250;
  original.camera_intrinsics.at(0)->set_value(changed);
  EXPECT_DOUBLE_EQ(snapshot.camera_intrinsics.at(0)->get_value()(0), 200);
  for (int epoch = 0; epoch < 3; ++epoch) {
    auto options = copy_ros2_reset_options(snapshot);
    EXPECT_TRUE(std::dynamic_pointer_cast<ov_core::CamEqui>(options.camera_intrinsics.at(1)));
    EXPECT_DOUBLE_EQ(options.camera_intrinsics.at(0)->get_value()(0), 200);
    options.camera_intrinsics.at(0)->set_value(changed);
    options.init_options.camera_intrinsics.at(1)->set_value(changed);
    EXPECT_DOUBLE_EQ(snapshot.camera_intrinsics.at(0)->get_value()(0), 200);
    EXPECT_DOUBLE_EQ(snapshot.init_options.camera_intrinsics.at(1)->get_value()(0), 200);
  }
}

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

TEST(ImuInterpolation, SkipsSamplesOnTheIntervalEnds) {
  // At small clock values neighbouring doubles are far closer than 1e-12 s apart
  const double start = 1.0, end = 1.01;
  auto selected = Propagator::select_imu_readings(
      {imu(0.995), imu(std::nextafter(start, 2.0), 1), imu(1.005, 2), imu(std::nextafter(end, 0.0), 3), imu(1.015)}, start, end, false);
  ASSERT_EQ(selected.size(), 3U);
  EXPECT_DOUBLE_EQ(selected[0].timestamp, start);
  EXPECT_DOUBLE_EQ(selected[1].timestamp, 1.005);
  EXPECT_DOUBLE_EQ(selected[2].timestamp, end);
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

TEST(EkfUpdate, MatchesDenseUpdateOverACorrelatedSubsetOfTheState) {
  auto estimate = state();
  for (double time : {1.0, 2.0}) {
    estimate->_timestamp = time;
    StateHelper::augment_clone(estimate, Eigen::Vector3d::Zero());
  }
  const int size = estimate->max_covariance_size();
  srand(3);
  const Eigen::MatrixXd root = Eigen::MatrixXd::Random(size, size);
  StateHelper::set_initial_covariance(estimate, root * root.transpose() + Eigen::MatrixXd::Identity(size, size),
                                      {estimate->_imu, estimate->_clones_IMU.at(1.0), estimate->_clones_IMU.at(2.0)});
  const Eigen::MatrixXd prior = StateHelper::get_full_covariance(estimate);
  const Eigen::VectorXd imu_before = estimate->_imu->value();

  // Measure the newest clone and the IMU, in that order, so the Jacobian columns are not in state order
  const std::vector<std::shared_ptr<ov_type::Type>> order = {estimate->_clones_IMU.at(2.0), estimate->_imu};
  const Eigen::MatrixXd H = Eigen::MatrixXd::Random(4, 21);
  const Eigen::VectorXd res = Eigen::VectorXd::Random(4);
  const Eigen::MatrixXd R = 0.5 * Eigen::MatrixXd::Identity(4, 4);
  StateHelper::EKFUpdate(estimate, order, H, res, R);

  Eigen::MatrixXd H_full = Eigen::MatrixXd::Zero(4, size);
  H_full.middleCols(order.at(0)->id(), 6) = H.leftCols(6);
  H_full.middleCols(order.at(1)->id(), 15) = H.rightCols(15);
  const Eigen::MatrixXd K = prior * H_full.transpose() * (H_full * prior * H_full.transpose() + R).inverse();
  const Eigen::MatrixXd posterior = prior - K * H_full * prior;
  EXPECT_LT((StateHelper::get_full_covariance(estimate) - posterior).norm(), 1e-9 * posterior.norm());
  // Position, velocity and biases update additively
  const Eigen::VectorXd dx = K * res;
  const Eigen::VectorXd expected = imu_before.tail(12) + dx.segment(estimate->_imu->id() + 3, 12);
  EXPECT_LT((estimate->_imu->value().bottomRows(12) - expected).norm(), 1e-9 * expected.norm());
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
