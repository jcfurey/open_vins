/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "AslDataset.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace ov_msckf;

namespace {

std::vector<std::string> split_csv(const std::string &line) {
  std::vector<std::string> fields;
  size_t start = 0;
  while (start <= line.size()) {
    size_t end = line.find(',', start);
    if (end == std::string::npos)
      end = line.size();
    std::string field = line.substr(start, end - start);
    const auto first = field.find_first_not_of(" \t\r");
    const auto last = field.find_last_not_of(" \t\r");
    fields.push_back(first == std::string::npos ? std::string() : field.substr(first, last - first + 1));
    start = end + 1;
  }
  return fields;
}

int64_t parse_ns(const std::string &text, const std::string &file) {
  errno = 0;
  char *end = nullptr;
  const long long value = std::strtoll(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0' || value < 0)
    throw std::runtime_error("Invalid nanosecond timestamp '" + text + "' in " + file);
  return value;
}

double parse_double(const std::string &text, const std::string &file) {
  char *end = nullptr;
  const double value = std::strtod(text.c_str(), &end);
  if (end == text.c_str() || *end != '\0' || !std::isfinite(value))
    throw std::runtime_error("Invalid number '" + text + "' in " + file);
  return value;
}

/// Calls row(fields) for each non-comment line.
template <typename Row> void read_csv(const std::string &file, Row row) {
  std::ifstream input(file);
  if (!input)
    throw std::runtime_error("Unable to open " + file);
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (line.empty() || line.front() == '#')
      continue;
    row(split_csv(line));
  }
}

struct CameraEntry {
  int64_t ns;
  std::string path;
};

} // namespace

double ov_msckf::asl_seconds(int64_t nanoseconds) {
  return static_cast<double>(nanoseconds / 1000000000) + static_cast<double>(nanoseconds % 1000000000) * 1e-9;
}

AslDataset::AslDataset(const std::string &root, int num_cameras, double sync_tolerance_s) {
  namespace fs = boost::filesystem;
  if (num_cameras < 1)
    throw std::invalid_argument("ASL datasets need at least one camera");
  folder_ = fs::is_directory(fs::path(root) / "mav0") ? (fs::path(root) / "mav0").string() : root;
  if (!fs::is_directory(fs::path(folder_) / "imu0"))
    throw std::runtime_error("No imu0/ folder under " + folder_ + " (expected an ASL/EuRoC dataset)");

  // Inertial readings, kept sorted and strictly increasing
  const std::string imu_file = (fs::path(folder_) / "imu0" / "data.csv").string();
  std::vector<std::pair<int64_t, ov_core::ImuData>> imu;
  read_csv(imu_file, [&](const std::vector<std::string> &fields) {
    if (fields.size() < 7)
      throw std::runtime_error("Expected 7 columns in " + imu_file);
    ov_core::ImuData sample;
    const int64_t ns = parse_ns(fields[0], imu_file);
    sample.timestamp = asl_seconds(ns);
    sample.wm << parse_double(fields[1], imu_file), parse_double(fields[2], imu_file), parse_double(fields[3], imu_file);
    sample.am << parse_double(fields[4], imu_file), parse_double(fields[5], imu_file), parse_double(fields[6], imu_file);
    imu.emplace_back(ns, sample);
  });
  std::stable_sort(imu.begin(), imu.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
  for (const auto &sample : imu) {
    if (!imu_ns_.empty() && sample.first <= imu_ns_.back())
      continue;
    imu_ns_.push_back(sample.first);
    imu_.push_back(sample.second);
  }
  if (imu_.empty())
    throw std::runtime_error("No IMU samples in " + imu_file);

  // Camera indices per sensor
  std::vector<std::vector<CameraEntry>> cameras(num_cameras);
  for (int id = 0; id < num_cameras; id++) {
    const fs::path camera = fs::path(folder_) / ("cam" + std::to_string(id));
    const std::string file = (camera / "data.csv").string();
    read_csv(file, [&](const std::vector<std::string> &fields) {
      if (fields.size() < 2 || fields[1].empty())
        throw std::runtime_error("Expected timestamp and filename columns in " + file);
      cameras[id].push_back({parse_ns(fields[0], file), (camera / "data" / fields[1]).string()});
    });
    std::stable_sort(cameras[id].begin(), cameras[id].end(), [](const auto &a, const auto &b) { return a.ns < b.ns; });
    if (cameras[id].empty())
      throw std::runtime_error("No images listed in " + file);
  }

  // Group frames whose timestamps agree across all cameras (two-pointer walk; inputs are sorted)
  const auto tolerance = static_cast<int64_t>(std::llround(sync_tolerance_s * 1e9));
  std::vector<size_t> cursor(num_cameras, 0);
  size_t matched_partners = 0;
  for (const auto &reference : cameras[0]) {
    Frame frame{reference.ns, {reference.path}};
    for (int id = 1; id < num_cameras; id++) {
      auto &i = cursor[id];
      const auto &list = cameras[id];
      while (i < list.size() && list[i].ns < reference.ns - tolerance)
        i++;
      if (i < list.size() && std::llabs(list[i].ns - reference.ns) <= tolerance)
        frame.paths.push_back(list[i++].path);
    }
    if ((int)frame.paths.size() == num_cameras) {
      if (frames_.empty() || frame.timestamp_ns > frames_.back().timestamp_ns) {
        matched_partners += num_cameras - 1;
        frames_.push_back(std::move(frame));
        continue;
      }
    }
    unmatched_++;
  }
  for (int id = 1; id < num_cameras; id++)
    unmatched_ += cameras[id].size();
  unmatched_ -= matched_partners;
}

ImagePrefetcher::ImagePrefetcher(std::vector<AslDataset::Frame> frames, size_t capacity)
    : frames_(std::move(frames)), capacity_(std::max<size_t>(1, capacity)) {
  worker_ = std::thread(&ImagePrefetcher::run, this);
}

ImagePrefetcher::~ImagePrefetcher() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  condition_.notify_all();
  if (worker_.joinable())
    worker_.join();
}

bool ImagePrefetcher::next(int64_t &timestamp_ns, std::vector<cv::Mat> &images) {
  std::unique_lock<std::mutex> lock(mutex_);
  condition_.wait(lock, [&] { return !ready_.empty() || done_; });
  if (ready_.empty()) {
    if (error_)
      std::rethrow_exception(error_);
    return false;
  }
  images = std::move(ready_.front());
  ready_.pop_front();
  timestamp_ns = frames_.at(consumed_++).timestamp_ns;
  lock.unlock();
  condition_.notify_all();
  return true;
}

void ImagePrefetcher::run() {
  try {
    for (const auto &frame : frames_) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] { return stop_ || ready_.size() < capacity_; });
        if (stop_)
          break;
      }
      std::vector<cv::Mat> images;
      for (const auto &path : frame.paths) {
        // Reduce 16-bit exports (e.g. TUM-VI *_16) exactly as cv_bridge converts mono16 to mono8 in the ROS runners
        cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
        if (image.empty())
          throw std::runtime_error("Unable to decode image " + path);
        if (image.depth() == CV_16U)
          image.convertTo(image, CV_8U, 255.0 / 65535.0);
        else if (image.depth() != CV_8U)
          throw std::runtime_error("Unsupported image depth in " + path);
        images.push_back(std::move(image));
      }
      std::lock_guard<std::mutex> lock(mutex_);
      ready_.push_back(std::move(images));
      condition_.notify_all();
    }
  } catch (...) {
    std::lock_guard<std::mutex> lock(mutex_);
    error_ = std::current_exception();
  }
  std::lock_guard<std::mutex> lock(mutex_);
  done_ = true;
  condition_.notify_all();
}
