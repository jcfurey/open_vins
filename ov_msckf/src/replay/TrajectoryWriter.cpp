/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "TrajectoryWriter.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <boost/filesystem.hpp>

#include "core/VioManager.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/IMU.h"

using namespace ov_msckf;

TrajectoryWriter::TrajectoryWriter(const std::string &path) {
  const auto parent = boost::filesystem::path(path).parent_path();
  if (!parent.empty())
    boost::filesystem::create_directories(parent);
  file_.open(path, std::ofstream::out | std::ofstream::trunc);
  if (!file_)
    throw std::runtime_error("Unable to open trajectory output: " + path);
  file_ << "# timestamp(s) tx ty tz qx qy qz qw Pr11 Pr12 Pr13 Pr22 Pr23 Pr33 Pt11 Pt12 Pt13 Pt22 Pt23 Pt33\n";
}

bool TrajectoryWriter::write(VioManager &app) {
  const auto state = app.get_state();
  if (!app.initialized() || state->_timestamp == last_state_time_)
    return false;
  last_state_time_ = state->_timestamp;

  // Report on the IMU clock; the state timestamp is the camera time of the last update.
  const double timestamp = state->_timestamp + state->_calib_dt_CAMtoIMU->value()(0);
  const Eigen::Vector4d q = state->_imu->quat(); // JPL q_GtoI has the same xyzw as Hamilton q_ItoG
  const Eigen::Vector3d p = state->_imu->pos();
  const std::vector<std::shared_ptr<ov_type::Type>> order = {state->_imu->q(), state->_imu->p()};
  const Eigen::MatrixXd cov = StateHelper::get_marginal_covariance(state, order);

  std::string line;
  auto append = [&line](const char *format, double value) {
    char buffer[400]; // %.9f of a diverged (but finite) double can exceed 300 characters
    std::snprintf(buffer, sizeof(buffer), format, value);
    line += buffer;
  };
  append("%.9f", timestamp);
  for (int i = 0; i < 3; i++)
    append(" %.9f", p(i));
  for (int i = 0; i < 4; i++)
    append(" %.9f", q(i));
  for (int block = 0; block < 2; block++) {
    for (int r = 0; r < 3; r++) {
      for (int c = r; c < 3; c++)
        append(" %.9e", cov(3 * block + r, 3 * block + c));
    }
  }
  file_ << line << '\n';
  if (!file_)
    throw std::runtime_error("Failed writing trajectory output");

  mix(timestamp);
  for (int i = 0; i < 3; i++)
    mix(p(i));
  for (int i = 0; i < 4; i++)
    mix(q(i));
  for (int r = 0; r < 6; r++) {
    for (int c = r; c < 6; c++)
      mix(cov(r, c));
  }
  if (count_++ == 0)
    first_time_ = timestamp;
  last_time_ = timestamp;
  return true;
}

std::string TrajectoryWriter::digest() const {
  char hex[17];
  std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(digest_));
  return hex;
}

void TrajectoryWriter::mix(double value) {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  for (int i = 0; i < 8; i++) {
    digest_ ^= (bits >> (8 * i)) & 0xffU;
    digest_ *= 1099511628211ULL;
  }
}
