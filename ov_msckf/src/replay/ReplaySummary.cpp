/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "ReplaySummary.h"

#include <sys/resource.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include <boost/filesystem.hpp>
#include <opencv2/imgproc.hpp>

#include "cam/CamBase.h"
#include "core/VioManagerOptions.h"

using namespace ov_msckf;

namespace {

std::string json_string(const std::string &text) {
  std::string out = "\"";
  for (const char c : text) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      char escaped[8];
      std::snprintf(escaped, sizeof(escaped), "\\u%04x", c);
      out += escaped;
    } else {
      out += c;
    }
  }
  return out + "\"";
}

std::string json_number(double value) {
  if (!std::isfinite(value))
    return "null";
  char text[64];
  std::snprintf(text, sizeof(text), "%.17g", value);
  return text;
}

double percentile(std::vector<double> values, double fraction) {
  if (values.empty())
    return std::nan("");
  const size_t index = std::min(values.size() - 1, static_cast<size_t>(std::ceil(fraction * values.size())) - (fraction > 0 ? 1 : 0));
  std::nth_element(values.begin(), values.begin() + index, values.end());
  return values[index];
}

} // namespace

std::map<int, cv::Mat> ov_msckf::replay_masks(const VioManagerOptions &params) {
  std::map<int, cv::Mat> masks;
  for (const auto &camera : params.camera_intrinsics) {
    const int factor = params.downsample_cameras ? 2 : 1;
    const cv::Size size(camera.second->w() * factor, camera.second->h() * factor);
    cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
    if (params.use_mask) {
      // Configured masks match the (possibly downsampled) calibration; the tracker downsamples masks with the images.
      const cv::Mat &configured = params.masks.at(camera.first);
      if (configured.size() == size)
        mask = configured;
      else
        cv::resize(configured, mask, size, 0.0, 0.0, cv::INTER_NEAREST);
    }
    masks.emplace(static_cast<int>(camera.first), mask);
  }
  return masks;
}

void ov_msckf::write_replay_summary(const std::string &path, const std::string &source, const SerialFeeder::Statistics &stats,
                                    const TrajectoryWriter &trajectory, const std::vector<double> &processing_seconds, double wall_seconds,
                                    const std::map<std::string, double> &extra) {
  struct rusage usage {};
  getrusage(RUSAGE_SELF, &usage);
  const double peak_rss_mib = static_cast<double>(usage.ru_maxrss) / 1024.0; // ru_maxrss is KiB on Linux
  const double cpu_seconds = usage.ru_utime.tv_sec + usage.ru_stime.tv_sec + 1e-6 * (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec);
  const double mean = processing_seconds.empty()
                          ? std::nan("")
                          : std::accumulate(processing_seconds.begin(), processing_seconds.end(), 0.0) / processing_seconds.size();
  const double max = processing_seconds.empty() ? std::nan("") : *std::max_element(processing_seconds.begin(), processing_seconds.end());

  const auto parent = boost::filesystem::path(path).parent_path();
  if (!parent.empty())
    boost::filesystem::create_directories(parent);
  std::ofstream out(path, std::ofstream::trunc);
  if (!out)
    throw std::runtime_error("Unable to write replay summary: " + path);
  out << "{\n";
  out << "  \"source\": " << json_string(source) << ",\n";
  out << "  \"imu\": {\"accepted\": " << stats.imu_accepted << ", \"rejected\": " << stats.imu_rejected << "},\n";
  out << "  \"cameras\": {\"accepted\": " << stats.cameras_accepted << ", \"rejected\": " << stats.cameras_rejected
      << ", \"rate_limited\": " << stats.cameras_rate_limited << ", \"processed\": " << stats.cameras_processed
      << ", \"unprocessed\": " << stats.cameras_unprocessed << "},\n";
  out << "  \"trajectory\": {\"poses\": " << trajectory.count() << ", \"first_time\": " << json_number(trajectory.first_time())
      << ", \"last_time\": " << json_number(trajectory.last_time()) << ", \"digest\": " << json_string(trajectory.digest()) << "},\n";
  out << "  \"timing\": {\"wall_s\": " << json_number(wall_seconds) << ", \"cpu_s\": " << json_number(cpu_seconds)
      << ", \"estimator_s\": " << json_number(stats.processing_seconds) << ", \"frame_ms_mean\": " << json_number(1e3 * mean)
      << ", \"frame_ms_median\": " << json_number(1e3 * percentile(processing_seconds, 0.5))
      << ", \"frame_ms_p95\": " << json_number(1e3 * percentile(processing_seconds, 0.95))
      << ", \"frame_ms_max\": " << json_number(1e3 * max) << "},\n";
  out << "  \"memory\": {\"peak_rss_mib\": " << json_number(peak_rss_mib) << "}";
  for (const auto &item : extra)
    out << ",\n  " << json_string(item.first) << ": " << json_number(item.second);
  out << "\n}\n";
  if (!out)
    throw std::runtime_error("Failed writing replay summary: " + path);
}
