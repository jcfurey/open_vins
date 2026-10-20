/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <gtest/gtest.h>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <random>
#include "core/VioManagerOptions.h"
#include "update/UpdaterHelper.h"
#include "update/VisualUpdate.h"
#include "state/State.h"
#include "cam/CamRadtan.h"

using namespace ov_msckf;
namespace {
Eigen::MatrixXd random_matrix(int rows, int cols) {
  static std::mt19937 random(77);
  std::normal_distribution<double> distribution;
  return Eigen::MatrixXd::NullaryExpr(rows, cols, [&] { return distribution(random); });
}
}

TEST(VisualBackends, AnisotropicWhiteningAndLandmarkEliminationMatchWeightedSchurComplement) {
  auto Hf = random_matrix(16, 3), Hx = random_matrix(16, 5);
  Eigen::VectorXd residual = random_matrix(16, 1);
  std::unordered_map<size_t, std::vector<double>> times{{0, {1, 2, 3, 4, 5, 6, 7, 8}}};
  ov_core::ObservationCovariances covariance;
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(16, 16);
  for (int i = 0; i < 8; ++i) {
    const Eigen::Matrix2d A = random_matrix(2, 2);
    const Eigen::Matrix2d block = A * A.transpose() + Eigen::Matrix2d::Identity();
    covariance[0][i + 1] = block;
    R.block<2, 2>(2 * i, 2 * i) = block;
  }
  const Eigen::MatrixXd W = R.llt().solve(Eigen::MatrixXd::Identity(16, 16));
  const Eigen::MatrixXd schur = W - W * Hf * (Hf.transpose() * W * Hf).ldlt().solve(Hf.transpose() * W);
  const Eigen::MatrixXd expected_information = Hx.transpose() * schur * Hx;
  const Eigen::VectorXd expected_gradient = Hx.transpose() * schur * residual;
  ASSERT_TRUE(VisualUpdate::whiten(covariance, times, Hf, Hx, residual));
  UpdaterHelper::nullspace_project_inplace(Hf, Hx, residual);
  EXPECT_TRUE((Hx.transpose() * Hx).isApprox(expected_information, 1e-10));
  EXPECT_TRUE((Hx.transpose() * residual).isApprox(expected_gradient, 1e-10));
}

TEST(VisualBackends, InvalidCovarianceRejectsTheFeatureWithoutPartialWhitening) {
  auto Hf = random_matrix(4, 3), Hx = random_matrix(4, 5);
  Eigen::VectorXd residual = random_matrix(4, 1);
  const auto before_f = Hf, before_x = Hx;
  const auto before_r = residual;
  std::unordered_map<size_t, std::vector<double>> times{{0, {1, 2}}};
  ov_core::ObservationCovariances covariance;
  covariance[0][1] = 5 * Eigen::Matrix2d::Identity();
  covariance[0][2] = -Eigen::Matrix2d::Identity();
  EXPECT_FALSE(VisualUpdate::whiten(covariance, times, Hf, Hx, residual));
  EXPECT_EQ(Hf, before_f);
  EXPECT_EQ(Hx, before_x);
  EXPECT_EQ(residual, before_r);
  covariance.clear();
  EXPECT_TRUE(VisualUpdate::whiten(covariance, times, Hf, Hx, residual));
  EXPECT_EQ(Hx, before_x);
}

TEST(VisualBackends, InnovationSpectrumMatchesTheFullCorrelatedSolve) {
  const auto A = random_matrix(11, 7);
  const Eigen::MatrixXd prior = A * A.transpose();
  const Eigen::VectorXd residual = random_matrix(11, 1);
  VisualUpdate::Innovation block;
  ASSERT_TRUE(VisualUpdate::innovation(prior, residual, 1.7, block));
  for (double scale : {1.0, 1.5, 4.0}) {
    const double expected = residual.dot((prior + 1.7 * scale * Eigen::MatrixXd::Identity(11, 11)).llt().solve(residual)) / 11;
    EXPECT_NEAR(block.normalized_nis(scale), expected, 1e-11);
  }
}

TEST(VisualBackends, CalibrationFindsTheSmallestBoundedNoiseInflation) {
  VisualUpdate::Innovation block;
  block.eigenvalues = Eigen::VectorXd::Zero(4);
  block.squared_residual = Eigen::VectorXd::Constant(4, 2.0);
  std::vector<VisualUpdate::Innovation> population(6, block);
  const double scale = VisualUpdate::minimum_inflation(population);
  EXPECT_NEAR(scale, 1.336, .002);
  EXPECT_GE(scale, 1.0);
  population.resize(5);
  EXPECT_EQ(VisualUpdate::minimum_inflation(population), 1.0);
  population.assign(6, block);
  for (auto &value : population)
    value.squared_residual *= 100;
  EXPECT_EQ(VisualUpdate::minimum_inflation(population), 4.0);
  for (auto &value : population)
    value.squared_residual.setZero();
  EXPECT_EQ(VisualUpdate::minimum_inflation(population), 1.0);
}

TEST(VisualBackends, Fej2RemovesOnlyTheErrorSubspaceAndCannotAddInformation) {
  const auto source = random_matrix(17, 2);
  const Eigen::MatrixXd difference = source * random_matrix(2, 6); // Intentionally rank two.
  Eigen::MatrixXd H = random_matrix(17, 12);
  H.leftCols(6) = difference;
  H.col(11).setZero(); // An unobservable gauge direction.
  Eigen::VectorXd residual = random_matrix(17, 1);
  const auto A = random_matrix(17, 17);
  Eigen::MatrixXd R = A * A.transpose() + Eigen::MatrixXd::Identity(17, 17);
  const Eigen::MatrixXd original_information = H.transpose() * R.llt().solve(H);
  ASSERT_EQ(VisualUpdate::fej2_project(H, residual, R, difference), 2);
  EXPECT_EQ(H.rows(), 15);
  EXPECT_LT(H.leftCols(6).norm(), 1e-10);
  EXPECT_LT(H.col(11).norm(), 1e-12);
  EXPECT_TRUE(R.isIdentity(1e-12));
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> loss(original_information - H.transpose() * H);
  EXPECT_GT(loss.eigenvalues().minCoeff(), -1e-10);
}

TEST(VisualBackends, ZeroFej2DifferencePreservesTheFullMeasurementAndFullRankLeavesNone) {
  Eigen::MatrixXd H = random_matrix(4, 5), R = 2 * Eigen::MatrixXd::Identity(4, 4);
  Eigen::VectorXd residual = random_matrix(4, 1);
  const auto original_H = H, original_R = R;
  const auto original_res = residual;
  EXPECT_EQ(VisualUpdate::fej2_project(H, residual, R, Eigen::MatrixXd::Zero(4, 6)), 0);
  EXPECT_EQ(H, original_H);
  EXPECT_EQ(R, original_R);
  EXPECT_EQ(residual, original_res);
  EXPECT_EQ(VisualUpdate::fej2_project(H, residual, R, Eigen::MatrixXd::Identity(4, 4)), 4);
  EXPECT_EQ(H.rows(), 0);
  EXPECT_EQ(residual.size(), 0);
  EXPECT_EQ(R.rows(), 0);
}

TEST(VisualBackends, InvalidModesAndIncompatibleFejSettingsAreRejected) {
  VioManagerOptions options;
  options.visual_update_backend = "rdic_fej2";
  options.configure_visual_update();
  EXPECT_TRUE(options.msckf_options.use_observation_covariance);
  EXPECT_FALSE(options.msckf_options.use_fej2);
  EXPECT_TRUE(options.slam_options.use_fej2);
  options.state_options.do_fej = false;
  EXPECT_THROW(options.configure_visual_update(), std::invalid_argument);
  options.visual_update_backend = "unknown";
  EXPECT_THROW(options.configure_visual_update(), std::invalid_argument);
}

TEST(VisualBackends, CurrentJacobianEvaluationDoesNotChangeConfiguredFejOrItsVariableOrder) {
  StateOptions options;
  options.num_cameras = 1;
  options.do_fej = true;
  auto state = std::make_shared<State>(options);
  auto camera = std::make_shared<ov_core::CamRadtan>(320, 240);
  Eigen::Matrix<double, 8, 1> intrinsics;
  intrinsics << 200, 200, 160, 120, 0, 0, 0, 0;
  camera->set_value(intrinsics);
  state->_cam_intrinsics_cameras[0] = camera;
  state->_cam_intrinsics.at(0)->set_value(intrinsics);
  Eigen::Matrix<double, 7, 1> pose;
  pose << 0, 0, 0, 1, 0, 0, 0;
  state->_calib_IMUtoCAM.at(0)->set_value(pose);
  auto clone = std::make_shared<ov_type::PoseJPL>();
  clone->set_fej(pose);
  pose.tail<3>() << .2, -.1, .1;
  clone->set_value(pose);
  state->_clones_IMU[1] = clone;
  UpdaterHelper::UpdaterHelperFeature feature;
  feature.feat_representation = ov_type::LandmarkRepresentation::Representation::GLOBAL_3D;
  feature.p_FinG << .5, .2, 4;
  feature.p_FinG_fej << .6, .2, 4.1;
  feature.timestamps[0] = {1};
  feature.uvs[0] = {Eigen::Vector2f(180, 130)};
  Eigen::MatrixXd feature_fej, feature_current, H_fej, H_current;
  Eigen::VectorXd r_fej, r_current;
  std::vector<std::shared_ptr<ov_type::Type>> order_fej, order_current;
  UpdaterHelper::get_feature_jacobian_full(state, feature, feature_fej, H_fej, r_fej, order_fej);
  UpdaterHelper::get_feature_jacobian_full(state, feature, feature_current, H_current, r_current, order_current, false);
  EXPECT_TRUE(state->_options.do_fej);
  EXPECT_EQ(order_fej, order_current);
  EXPECT_EQ(r_fej, r_current);
  EXPECT_GT((H_fej - H_current).norm(), .1);
}
