/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_MSCKF_UPDATER_OPTIONS_H
#define OV_MSCKF_UPDATER_OPTIONS_H

#include "utils/print.h"

namespace ov_msckf {

/**
 * @brief Struct which stores general updater options
 */
struct UpdaterOptions {

  /// What chi-squared multipler we should apply
  double chi2_multipler = 5;

  /// Noise sigma for our raw pixel measurements
  double sigma_pix = 1;

  /// Covariance for our raw pixel measurements
  double sigma_pix_sq = 1;

  bool use_observation_covariance = false;
  bool calibrate_innovation = false;
  bool use_fej2 = false;
  double innovation_max_variance = 4.0;
  double innovation_quantile = .8;
  int innovation_min_blocks = 6;
  double fej2_rank_tolerance = 1e-6;

  /// Nice print function of what parameters we have loaded
  void print() {
    PRINT_DEBUG("    - chi2_multipler: %.1f\n", chi2_multipler);
    PRINT_DEBUG("    - sigma_pix: %.2f\n", sigma_pix);
    PRINT_DEBUG("    - observation covariance: %d, batch calibration: %d, FEJ2: %d\n",
                use_observation_covariance, calibrate_innovation, use_fej2);
  }
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_OPTIONS_H
