/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef OV_MSCKF_TRAJECTORY_WRITER_H
#define OV_MSCKF_TRAJECTORY_WRITER_H

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>

namespace ov_msckf {

class VioManager;

/**
 * @brief Writes estimated IMU poses in the ov_eval trajectory format.
 *
 * Each row is `timestamp(s) tx ty tz qx qy qz qw Pr11 Pr12 Pr13 Pr22 Pr23 Pr33 Pt11 Pt12 Pt13 Pt22 Pt23 Pt33` with the
 * timestamp on the IMU clock, the Hamilton quaternion of the IMU in the global frame, and the upper triangles of the
 * orientation (local IMU-frame error, as ov_eval expects) and position covariances. A digest of the full-precision
 * values is kept so two runs can be compared bit-for-bit.
 */
class TrajectoryWriter {
public:
  /// Opens (truncates) the output file, creating parent directories. Throws on failure.
  explicit TrajectoryWriter(const std::string &path);

  /// Append the current pose if the estimator is initialized and has advanced to a new state time.
  bool write(VioManager &app);

  size_t count() const { return count_; }
  double first_time() const { return first_time_; }
  double last_time() const { return last_time_; }

  /// FNV-1a digest over the exact bits of every written value (hex string).
  std::string digest() const;

private:
  void mix(double value);

  std::ofstream file_;
  size_t count_ = 0;
  double first_time_ = -1.0, last_time_ = -1.0, last_state_time_ = -1.0;
  uint64_t digest_ = 1469598103934665603ULL;
};

} // namespace ov_msckf

#endif // OV_MSCKF_TRAJECTORY_WRITER_H
