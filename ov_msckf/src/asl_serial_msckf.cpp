/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include "cam/CamBase.h"
#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "replay/AslDataset.h"
#include "replay/ReplaySummary.h"
#include "replay/SerialFeeder.h"
#include "replay/TrajectoryWriter.h"
#include "utils/colors.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"

using namespace ov_msckf;

namespace {

const char *usage = "usage: asl_serial_msckf <estimator_config.yaml> <dataset> <output_dir> [--start S] [--duration S] [--prefetch N]\n"
                    "  dataset     ASL/EuRoC sequence folder (containing mav0/) or the mav0/ folder itself\n"
                    "  output_dir  receives trajectory.txt (ov_eval format), timing.csv and summary.json\n"
                    "  --start     seconds to skip from the first IMU sample (default 0)\n"
                    "  --duration  seconds to process after the start (default: all)\n"
                    "  --prefetch  decoded frames to buffer ahead of the estimator (default 8)\n"
                    "  --verbosity overrides the configured print level (ALL, DEBUG, INFO, WARNING, ERROR, SILENT)\n";

struct Arguments {
  std::string config, dataset, output;
  std::string verbosity;
  double start = 0.0, duration = -1.0;
  size_t prefetch = 8;
};

Arguments parse_arguments(int argc, char **argv) {
  Arguments args;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; i++) {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc)
        throw std::invalid_argument(arg + " requires a value");
      return argv[++i];
    };
    if (arg == "--start")
      args.start = std::stod(value());
    else if (arg == "--duration")
      args.duration = std::stod(value());
    else if (arg == "--prefetch")
      args.prefetch = std::stoul(value());
    else if (arg == "--verbosity")
      args.verbosity = value();
    else if (arg == "-h" || arg == "--help")
      throw std::invalid_argument("");
    else
      positional.push_back(arg);
  }
  if (positional.size() != 3 || !std::isfinite(args.start) || args.start < 0 || !std::isfinite(args.duration))
    throw std::invalid_argument("expected a config, a dataset, and an output folder");
  args.config = positional[0];
  args.dataset = positional[1];
  args.output = positional[2];
  return args;
}

} // namespace

int main(int argc, char **argv) {
  Arguments args;
  try {
    args = parse_arguments(argc, argv);
  } catch (const std::exception &error) {
    if (*error.what())
      std::cerr << "asl_serial_msckf: " << error.what() << "\n";
    std::cerr << usage;
    return EXIT_FAILURE;
  }

  try {
    auto parser = std::make_shared<ov_core::YamlParser>(args.config);
    std::string verbosity = "INFO";
    parser->parse_config("verbosity", verbosity);
    ov_core::Printer::setPrintLevel(args.verbosity.empty() ? verbosity : args.verbosity);

    VioManagerOptions params;
    params.print_and_load(parser);
    if (make_replay_deterministic(params))
      PRINT_INFO("[REPLAY]: dynamic initialization runs single-threaded without its time budget for reproducibility\n");
    if (!parser->successful())
      throw std::invalid_argument("Unable to parse required OpenVINS parameters");
    const int num_cameras = params.state_options.num_cameras;
    if (num_cameras < 1 || !std::isfinite(params.track_frequency) || params.track_frequency <= 0)
      throw std::invalid_argument("Invalid camera count or tracking rate");

    const AslDataset dataset(args.dataset, num_cameras);
    const auto &imu = dataset.imu();
    const auto &imu_ns = dataset.imu_timestamps_ns();
    const int64_t begin_ns = imu_ns.front() + std::llround(args.start * 1e9);
    const int64_t end_ns = args.duration < 0 ? std::numeric_limits<int64_t>::max() : begin_ns + std::llround(args.duration * 1e9);
    std::vector<AslDataset::Frame> frames;
    for (const auto &frame : dataset.frames()) {
      if (frame.timestamp_ns >= begin_ns && frame.timestamp_ns <= end_ns)
        frames.push_back(frame);
    }
    PRINT_INFO("[REPLAY]: %zu IMU samples, %zu of %zu synchronized frames selected, %zu unmatched images\n", imu.size(), frames.size(),
               dataset.frames().size(), dataset.unmatched_frames());

    const auto masks = replay_masks(params);
    auto app = std::make_shared<VioManager>(params);
    TrajectoryWriter trajectory((boost::filesystem::path(args.output) / "trajectory.txt").string());
    std::ofstream timing((boost::filesystem::path(args.output) / "timing.csv").string(), std::ofstream::trunc);
    if (!timing)
      throw std::runtime_error("Unable to write timing.csv in " + args.output);
    timing << "# camera timestamp(s),estimator seconds,initialized\n";
    std::vector<double> processing;
    processing.reserve(frames.size());
    SerialFeeder feeder(app, [&](const ov_core::CameraData &packet, double seconds) {
      trajectory.write(*app);
      processing.push_back(seconds);
      char row[96];
      std::snprintf(row, sizeof(row), "%.9f,%.6f,%d\n", packet.timestamp, seconds, app->initialized() ? 1 : 0);
      timing << row;
    });

    const auto wall_start = std::chrono::steady_clock::now();
    ImagePrefetcher images(frames, args.prefetch);
    size_t next_imu = std::lower_bound(imu_ns.begin(), imu_ns.end(), begin_ns) - imu_ns.begin();
    int64_t frame_ns = 0;
    std::vector<cv::Mat> frame_images;
    bool have_frame = images.next(frame_ns, frame_images);
    while (true) {
      // IMU first on ties, matching sensor arrival order in recorded logs
      if (next_imu < imu.size() && imu_ns[next_imu] <= end_ns && (!have_frame || imu_ns[next_imu] <= frame_ns)) {
        feeder.feed_imu(imu[next_imu++]);
        continue;
      }
      if (!have_frame)
        break;
      ov_core::CameraData packet;
      packet.timestamp = asl_seconds(frame_ns);
      for (int id = 0; id < num_cameras; id++) {
        if (frame_images.at(id).size() != masks.at(id).size())
          throw std::runtime_error("cam" + std::to_string(id) + " image size does not match its calibration");
        packet.sensor_ids.push_back(id);
        packet.images.push_back(frame_images.at(id));
        packet.masks.push_back(masks.at(id));
      }
      feeder.feed_camera(std::move(packet));
      have_frame = images.next(frame_ns, frame_images);
    }
    feeder.finish();
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();

    const auto &stats = feeder.statistics();
    write_replay_summary((boost::filesystem::path(args.output) / "summary.json").string(), dataset.folder(), stats, trajectory, processing,
                         wall, {{"unmatched_images", static_cast<double>(dataset.unmatched_frames())}});
    PRINT_INFO("[REPLAY]: processed %zu frames (%zu unprocessed, %zu rate limited) in %.2f s; wrote %zu poses, digest %s\n",
               stats.cameras_processed, stats.cameras_unprocessed, stats.cameras_rate_limited, wall, trajectory.count(),
               trajectory.digest().c_str());
    if (trajectory.count() == 0) {
      PRINT_ERROR(RED "[REPLAY]: the estimator never initialized; no trajectory was produced\n" RESET);
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "asl_serial_msckf failed: " << error.what() << std::endl;
    return EXIT_FAILURE;
  }
}
