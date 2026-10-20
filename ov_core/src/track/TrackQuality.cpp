/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "TrackQuality.h"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

using namespace ov_core;
namespace {
double median(std::vector<double> values) {
  if (values.empty())
    return 0;
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  return n % 2 ? values[n / 2] : .5 * (values[n / 2 - 1] + values[n / 2]);
}
std::array<double, 5> values(const FlowQuality &q) { return {q.temporal, q.balance, q.stereo, q.backward, q.disparity}; }
bool inside(const cv::Point2f &p, const cv::Mat &image, int margin) {
  return std::isfinite(p.x) && std::isfinite(p.y) && p.x >= margin && p.y >= margin &&
         p.x < image.cols - margin && p.y < image.rows - margin;
}
} // namespace

std::vector<Eigen::Matrix2d> TrackQuality::covariance(const cv::Mat &previous, const cv::Mat &current,
                                                    const std::vector<cv::Point2f> &old_points,
                                                    const std::vector<cv::Point2f> &new_points,
                                                    const std::vector<FlowQuality> &diagnostics,
                                                    const std::vector<size_t> &ages) {
  const size_t count = new_points.size();
  if (old_points.size() != count || diagnostics.size() != count || ages.size() != count || previous.type() != CV_8UC1 ||
      current.type() != CV_8UC1)
    throw std::invalid_argument("Invalid tracking-quality observation/image dimensions");
  std::vector<Eigen::Matrix2d> result(count, Eigen::Matrix2d::Identity());
  if (count == 0)
    return result;
  std::array<double, 5> center, scale;
  const std::array<double, 5> floors{1.0, .5, .5, .5, 1.0}, weights{1.0, .5, 1.0, .75, .5};
  for (size_t family = 0; family < 5; ++family) {
    std::vector<double> population;
    for (const auto &q : diagnostics)
      if (std::isfinite(values(q)[family]))
        population.push_back(values(q)[family]);
    center[family] = median(population);
    for (auto &value : population)
      value = std::abs(value - center[family]);
    scale[family] = std::max(floors[family], 1.4826 * median(population));
  }
  cv::Mat gradient_x, gradient_y;
  cv::Sobel(current, gradient_x, CV_32F, 1, 0, 3, 1.0 / 8);
  cv::Sobel(current, gradient_y, CV_32F, 0, 1, 3, 1.0 / 8);
  std::vector<Eigen::Matrix2d> information(count, Eigen::Matrix2d::Zero());
  std::vector<double> eigen_min;
  std::vector<double> quality(count, 0.0);
  std::vector<bool> valid(count, false);
  for (size_t i = 0; i < count; ++i) {
    double risk = 0;
    const auto diagnostic = values(diagnostics[i]);
    for (size_t family = 0; family < 5; ++family) {
      if (!std::isfinite(diagnostic[family]))
        continue;
      const double z = std::max(0.0, (diagnostic[family] - center[family]) / scale[family]);
      risk += weights[family] * z * z;
    }
    quality[i] = std::exp(-risk / (1.0 + .25 * std::log1p(ages[i])));
    const double sigma = 1.0 + 4.0 * (1.0 - quality[i]);
    result[i] *= sigma * sigma;
    if (!inside(old_points[i], previous, 8) || !inside(new_points[i], current, 8)) {
      result[i] = 25.0 * Eigen::Matrix2d::Identity();
      continue;
    }
    cv::Mat a, b, gx, gy;
    cv::getRectSubPix(previous, cv::Size(15, 15), old_points[i], a, CV_32F);
    cv::getRectSubPix(current, cv::Size(15, 15), new_points[i], b, CV_32F);
    cv::getRectSubPix(gradient_x, cv::Size(15, 15), new_points[i], gx);
    cv::getRectSubPix(gradient_y, cv::Size(15, 15), new_points[i], gy);
    const double brightness_change = cv::mean(b)[0] - cv::mean(a)[0];
    for (int row = 0; row < 15; ++row) {
      for (int col = 0; col < 15; ++col) {
        const double residual = b.at<float>(row, col) - a.at<float>(row, col) - brightness_change;
        const double huber = std::abs(residual) <= 8.0 ? 1.0 : 8.0 / std::abs(residual);
        const Eigen::Vector2d gradient(gx.at<float>(row, col), gy.at<float>(row, col));
        information[i].noalias() += (huber / 225.0) * gradient * gradient.transpose();
      }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen(information[i]);
    valid[i] = eigen.info() == Eigen::Success && eigen.eigenvalues().allFinite();
    if (valid[i])
      eigen_min.push_back(eigen.eigenvalues()[0]);
  }
  const double reference = std::clamp(median(eigen_min), 5.0, 5000.0);
  for (size_t i = 0; i < count; ++i) {
    // Directional information is unreliable for a new or very poor correspondence.
    if (!valid[i] || ages[i] < 2 || quality[i] < .25 ||
        (std::isfinite(diagnostics[i].stereo) && diagnostics[i].stereo > 2.0))
      continue;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen(information[i]);
    if (eigen.eigenvalues().maxCoeff() < 1e-6) {
      result[i] += 24.0 * Eigen::Matrix2d::Identity();
      continue;
    }
    const double weak = std::clamp(std::sqrt(reference / std::max(1e-6, eigen.eigenvalues()[0])), 1.0, 25.0);
    const double strong = std::min(weak, std::clamp(std::sqrt(reference / std::max(1e-6, eigen.eigenvalues()[1])), 1.0, 6.0));
    result[i].noalias() += (weak - 1) * eigen.eigenvectors().col(0) * eigen.eigenvectors().col(0).transpose();
    result[i].noalias() += (strong - 1) * eigen.eigenvectors().col(1) * eigen.eigenvectors().col(1).transpose();
  }
  return result;
}

double TrackQuality::epipolar_error(const Eigen::Matrix3d &essential, const Eigen::Vector3d &left,
                                  const Eigen::Vector3d &right, double focal) {
  const Eigen::Vector3d a = essential * left, b = essential.transpose() * right;
  const double denominator = a.head<2>().squaredNorm() + b.head<2>().squaredNorm();
  if (denominator < 1e-20)
    return std::numeric_limits<double>::quiet_NaN();
  return focal * std::abs(right.dot(a)) / std::sqrt(denominator);
}
