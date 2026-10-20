/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "VisualUpdate.h"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <algorithm>
#include <boost/math/distributions/chi_squared.hpp>
#include <cmath>
#include <stdexcept>

using namespace ov_msckf;

bool VisualUpdate::whiten(const ov_core::ObservationCovariances &covariances,
                         const std::unordered_map<size_t, std::vector<double>> &times,
                         Eigen::MatrixXd &Hf, Eigen::MatrixXd &Hx, Eigen::VectorXd &res, Eigen::MatrixXd *extra) {
  Eigen::Index rows = 0;
  for (const auto &camera : times)
    rows += 2 * camera.second.size();
  if (rows != Hf.rows() || rows != Hx.rows() || rows != res.rows() || (extra && rows != extra->rows()))
    throw std::invalid_argument("Visual covariance/Jacobian observation counts differ");
  // Validate the complete input before modifying any blocks.
  for (const auto &camera : times) {
    auto found = covariances.find(camera.first);
    if (found == covariances.end())
      continue;
    for (double time : camera.second) {
      auto observation = found->second.find(time);
      if (observation == found->second.end())
        continue;
      const auto &R = observation->second;
      if (!R.allFinite() || !R.isApprox(R.transpose(), 1e-10) || Eigen::LLT<Eigen::Matrix2d>(R).info() != Eigen::Success)
        return false;
    }
  }
  Eigen::Index row = 0;
  for (const auto &camera : times) {
    auto found = covariances.find(camera.first);
    for (double time : camera.second) {
      if (found != covariances.end() && found->second.count(time)) {
        Eigen::LLT<Eigen::Matrix2d> factor(found->second.at(time));
        Hf.middleRows(row, 2) = factor.matrixL().solve(Hf.middleRows(row, 2)).eval();
        Hx.middleRows(row, 2) = factor.matrixL().solve(Hx.middleRows(row, 2)).eval();
        res.segment(row, 2) = factor.matrixL().solve(res.segment(row, 2)).eval();
        if (extra)
          extra->middleRows(row, 2) = factor.matrixL().solve(extra->middleRows(row, 2)).eval();
      }
      row += 2;
    }
  }
  return Hf.allFinite() && Hx.allFinite() && res.allFinite() && (!extra || extra->allFinite());
}

double VisualUpdate::Innovation::normalized_nis(double variance_scale) const {
  return (squared_residual.array() / (eigenvalues.array() + variance_scale)).sum() / eigenvalues.size();
}

bool VisualUpdate::innovation(const Eigen::MatrixXd &projected_prior, const Eigen::VectorXd &res, double noise_variance,
                              Innovation &result) {
  if (res.size() == 0 || projected_prior.rows() != res.size() || projected_prior.cols() != res.size() ||
      !projected_prior.allFinite() || !res.allFinite() || !std::isfinite(noise_variance) || noise_variance <= 0)
    return false;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(projected_prior / noise_variance);
  if (eigen.info() != Eigen::Success)
    return false;
  // Only tolerate rounding of a positive semidefinite projected prior.
  if (eigen.eigenvalues().minCoeff() < -1e-8 * std::max(1.0, eigen.eigenvalues().cwiseAbs().maxCoeff()))
    return false;
  result.eigenvalues = eigen.eigenvalues().cwiseMax(0);
  result.squared_residual = (eigen.eigenvectors().transpose() * res).array().square() / noise_variance;
  return result.squared_residual.allFinite();
}

namespace {
double empirical_quantile(std::vector<double> values, double quantile) {
  std::sort(values.begin(), values.end());
  const double index = quantile * (values.size() - 1);
  const size_t lower = static_cast<size_t>(index), upper = std::min(lower + 1, values.size() - 1);
  return values[lower] + (index - lower) * (values[upper] - values[lower]);
}
} // namespace

double VisualUpdate::minimum_inflation(const std::vector<Innovation> &population, double maximum, double quantile,
                                      size_t minimum_blocks) {
  if (!std::isfinite(maximum) || maximum < 1 || !(quantile > 0 && quantile < 1) || minimum_blocks == 0)
    throw std::invalid_argument("Invalid RDIC innovation calibration bounds");
  if (population.size() < minimum_blocks)
    return 1.0;
  std::vector<double> dimensions;
  for (const auto &block : population) {
    if (block.eigenvalues.size() == 0 || block.eigenvalues.size() != block.squared_residual.size() ||
        !block.eigenvalues.allFinite() || !block.squared_residual.allFinite() || block.eigenvalues.minCoeff() < 0 ||
        block.squared_residual.minCoeff() < 0)
      throw std::invalid_argument("Invalid RDIC innovation spectrum");
    dimensions.push_back(block.eigenvalues.size());
  }
  const int dof = std::max(1, static_cast<int>(empirical_quantile(dimensions, .5)));
  const double target = boost::math::quantile(boost::math::chi_squared(dof), quantile) / dof;
  auto statistic = [&](double scale) {
    std::vector<double> values;
    values.reserve(population.size());
    for (const auto &block : population)
      values.push_back(block.normalized_nis(scale));
    return empirical_quantile(std::move(values), quantile);
  };
  if (statistic(1.0) <= target)
    return 1.0;
  if (statistic(maximum) > target)
    return maximum;
  double lower = 1.0, upper = maximum;
  for (int iteration = 0; iteration < 16 && upper - lower > 1e-3; ++iteration) {
    const double middle = .5 * (lower + upper);
    if (statistic(middle) > target)
      lower = middle;
    else
      upper = middle;
  }
  return upper;
}

int VisualUpdate::fej2_project(Eigen::MatrixXd &Hx, Eigen::VectorXd &res, Eigen::MatrixXd &noise,
                              const Eigen::MatrixXd &pose_difference, double relative_tolerance, double absolute_tolerance) {
  if (Hx.rows() != res.size() || noise.rows() != res.size() || noise.cols() != res.size() ||
      pose_difference.rows() != res.size() || !Hx.allFinite() || !res.allFinite() || !noise.allFinite() ||
      !pose_difference.allFinite() || !(relative_tolerance > 0 && relative_tolerance < 1) || !(absolute_tolerance > 0))
    throw std::invalid_argument("Invalid FEJ2 projected system");
  if (res.size() == 0 || pose_difference.cols() == 0)
    return 0;
  Eigen::LLT<Eigen::MatrixXd> factor(noise);
  if (factor.info() != Eigen::Success)
    throw std::invalid_argument("FEJ2 requires positive definite measurement noise");
  Eigen::MatrixXd difference = factor.matrixL().solve(pose_difference);
  if (difference.cwiseAbs().maxCoeff() <= absolute_tolerance)
    return 0;
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(difference);
  qr.setThreshold(relative_tolerance);
  const int rank = qr.rank();
  if (rank == 0)
    return 0;
  Eigen::MatrixXd whitened_H = factor.matrixL().solve(Hx);
  Eigen::VectorXd whitened_res = factor.matrixL().solve(res);
  whitened_H = (qr.householderQ().adjoint() * whitened_H).eval();
  whitened_res = (qr.householderQ().adjoint() * whitened_res).eval();
  Hx = whitened_H.bottomRows(Hx.rows() - rank).eval();
  res = whitened_res.tail(res.rows() - rank).eval();
  noise = Eigen::MatrixXd::Identity(res.size(), res.size());
  return rank;
}
