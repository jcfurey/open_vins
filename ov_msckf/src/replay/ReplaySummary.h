/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef OV_MSCKF_REPLAY_SUMMARY_H
#define OV_MSCKF_REPLAY_SUMMARY_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "replay/SerialFeeder.h"
#include "replay/TrajectoryWriter.h"

namespace ov_msckf {

struct VioManagerOptions;

/// Per-camera masks for replayed frames: the configured mask, or an all-zero mask at the input resolution.
std::map<int, cv::Mat> replay_masks(const VioManagerOptions &params);

/// Write a machine-readable summary of a deterministic replay (counts, timing, memory, trajectory digest).
void write_replay_summary(const std::string &path, const std::string &source, const SerialFeeder::Statistics &stats,
                          const TrajectoryWriter &trajectory, const std::vector<double> &processing_seconds, double wall_seconds,
                          const std::map<std::string, double> &extra = {});

} // namespace ov_msckf

#endif // OV_MSCKF_REPLAY_SUMMARY_H
