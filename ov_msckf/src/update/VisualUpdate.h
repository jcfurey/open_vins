/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef OV_MSCKF_VISUAL_UPDATE_H
#define OV_MSCKF_VISUAL_UPDATE_H

#include "feat/ObservationCovariance.h"
#include <Eigen/Core>
#include <vector>

namespace ov_msckf {

/// Optional visual-update operations. See RDIC-MSCKF (2026), equations 17--35,
/// and FEJ2 technical report RPNG-2021-FEJ2, sections 2.2 and 2.4.2.
class VisualUpdate {
public:
  /// Whiten relative 2x2 pixel covariance BEFORE eliminating the landmark/bearing.
  /// The remaining noise is sigma_pix^2 I. Invalid covariance rejects the feature.
  static bool whiten(const ov_core::ObservationCovariances &covariances,
                     const std::unordered_map<size_t, std::vector<double>> &times,
                     Eigen::MatrixXd &Hf, Eigen::MatrixXd &Hx, Eigen::VectorXd &res,
                     Eigen::MatrixXd *extra = nullptr);

  struct Innovation {
    Eigen::VectorXd eigenvalues;
    Eigen::VectorXd squared_residual;
    double normalized_nis(double variance_scale) const;
  };
  /// Normalized innovation spectrum BEFORE feature gating, not the truncated accepted population.
  static bool innovation(const Eigen::MatrixXd &projected_prior, const Eigen::VectorXd &res, double noise_variance,
                         Innovation &result);
  static double minimum_inflation(const std::vector<Innovation> &population, double maximum = 4.0,
                                  double quantile = .8, size_t minimum_blocks = 6);

  /// Noise-whiten and project onto the left nullspace of the pose-error Jacobian.
  /// Returns removed rank; no remaining rows means no usable update, never a fallback to FEJ.
  static int fej2_project(Eigen::MatrixXd &Hx, Eigen::VectorXd &res, Eigen::MatrixXd &noise,
                          const Eigen::MatrixXd &pose_difference, double relative_tolerance = 1e-6,
                          double absolute_tolerance = 1e-8);
};
} // namespace ov_msckf
#endif
