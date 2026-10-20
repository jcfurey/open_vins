/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <gtest/gtest.h>
#include <random>

#include "update/UpdaterHelper.h"

using ov_msckf::UpdaterHelper;

namespace {
Eigen::MatrixXd random_matrix(int rows, int cols, unsigned seed = 17) {
  std::mt19937 generator(seed);
  std::normal_distribution<double> normal;
  Eigen::MatrixXd matrix(rows, cols);
  for (int col = 0; col < cols; ++col)
    for (int row = 0; row < rows; ++row)
      matrix(row, col) = normal(generator);
  return matrix;
}

void preserves_information(const Eigen::MatrixXd &input, const Eigen::VectorXd &residual) {
  Eigen::MatrixXd compressed = input;
  Eigen::VectorXd transformed = residual;
  UpdaterHelper::measurement_compress_inplace(compressed, transformed);
  EXPECT_EQ(compressed.rows(), std::min(input.rows(), input.cols()));
  EXPECT_EQ(compressed.cols(), input.cols());
  EXPECT_EQ(transformed.rows(), compressed.rows());
  const Eigen::MatrixXd information = input.transpose() * input;
  const Eigen::VectorXd gradient = input.transpose() * residual;
  EXPECT_LE((compressed.transpose() * compressed - information).norm(), 1e-10 * std::max(1., information.norm()));
  EXPECT_LE((compressed.transpose() * transformed - gradient).norm(), 1e-10 * std::max(1., gradient.norm()));
}
} // namespace

TEST(MeasurementCompression, PreservesInformationForTallVisualSystems) {
  for (const auto &shape : {std::pair<int, int>{37, 15}, {500, 120}, {1600, 240}}) {
    SCOPED_TRACE(std::to_string(shape.first) + "x" + std::to_string(shape.second));
    preserves_information(random_matrix(shape.first, shape.second), random_matrix(shape.first, 1, 23));
  }
}

TEST(MeasurementCompression, HandlesRankDeficientAndZeroJacobians) {
  Eigen::MatrixXd matrix = random_matrix(91, 20);
  matrix.rightCols(10) = matrix.leftCols(10);
  preserves_information(matrix, random_matrix(91, 1, 23));
  preserves_information(Eigen::MatrixXd::Zero(91, 20), random_matrix(91, 1, 23));
  preserves_information(Eigen::MatrixXd(12, 0), random_matrix(12, 1, 23));
}

TEST(MeasurementCompression, LeavesSquareAndWideSystemsUnchanged) {
  for (int cols : {10, 30}) {
    const Eigen::MatrixXd input = random_matrix(10, cols);
    const Eigen::VectorXd residual = random_matrix(10, 1, 23);
    Eigen::MatrixXd matrix = input;
    Eigen::VectorXd transformed = residual;
    UpdaterHelper::measurement_compress_inplace(matrix, transformed);
    EXPECT_EQ((matrix - input).norm(), 0.);
    EXPECT_EQ((transformed - residual).norm(), 0.);
  }
}

TEST(MeasurementCompression, MatchesTheUncompressedKalmanPosterior) {
  const Eigen::MatrixXd H = random_matrix(80, 15);
  const Eigen::VectorXd residual = random_matrix(80, 1, 23);
  const Eigen::MatrixXd root = random_matrix(15, 15, 31);
  const Eigen::MatrixXd prior = root * root.transpose() + Eigen::MatrixXd::Identity(15, 15);
  const double variance = .7;
  const Eigen::MatrixXd covariance = H * prior * H.transpose() + variance * Eigen::MatrixXd::Identity(80, 80);
  const Eigen::MatrixXd gain = covariance.llt().solve((prior * H.transpose()).transpose()).transpose();
  const Eigen::MatrixXd expected_covariance = prior - gain * H * prior;
  const Eigen::VectorXd expected_mean = gain * residual;
  Eigen::MatrixXd compressed = H;
  Eigen::VectorXd transformed = residual;
  UpdaterHelper::measurement_compress_inplace(compressed, transformed);
  const Eigen::MatrixXd compressed_covariance = compressed * prior * compressed.transpose() + variance * Eigen::MatrixXd::Identity(15, 15);
  const Eigen::MatrixXd compressed_gain = compressed_covariance.llt().solve((prior * compressed.transpose()).transpose()).transpose();
  EXPECT_LT((prior - compressed_gain * compressed * prior - expected_covariance).norm(), 1e-10 * prior.norm());
  EXPECT_LT((compressed_gain * transformed - expected_mean).norm(), 1e-10 * std::max(1., expected_mean.norm()));
}
