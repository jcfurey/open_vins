/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef OV_CORE_OBSERVATION_COVARIANCE_H
#define OV_CORE_OBSERVATION_COVARIANCE_H

#include <Eigen/Core>
#include <map>
#include <unordered_map>

namespace ov_core {
// Dimensionless pixel-plane covariance multipliers, keyed by camera and acquisition time.
// Actual pixel covariance is sigma_pix^2 times this matrix. Missing metadata means identity.
using ObservationCovariances = std::unordered_map<size_t, std::map<double, Eigen::Matrix2d>>;
} // namespace ov_core
#endif
