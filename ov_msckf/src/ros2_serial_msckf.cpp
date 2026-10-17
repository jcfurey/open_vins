/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "replay/FrameSynchronizer.h"
#include "replay/ReplaySummary.h"
#include "replay/SerialFeeder.h"
#include "replay/TrajectoryWriter.h"
#include "ros/ROS2Visualizer.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include "utils/ros2_parameters.h"

using namespace ov_msckf;

namespace {

double stamp_seconds(const builtin_interfaces::msg::Time &stamp) {
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

template <typename Message> Message deserialize(const rosbag2_storage::SerializedBagMessage &bag_message) {
  static rclcpp::Serialization<Message> serialization;
  rclcpp::SerializedMessage serialized(*bag_message.serialized_data);
  Message message;
  serialization.deserialize_message(&serialized, &message);
  return message;
}

} // namespace

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  int result = EXIT_SUCCESS;
  try {
    rclcpp::NodeOptions options;
    options.allow_undeclared_parameters(false);
    options.automatically_declare_parameters_from_overrides(false);
    auto node = std::make_shared<rclcpp::Node>("ros2_serial_msckf", options);
    const auto positional = rclcpp::remove_ros_arguments(argc, argv);
    std::string config_path = positional.size() > 1 ? positional[1] : "unset_path_to_config.yaml";
    config_path = ov_core::startup_parameter<std::string>(*node, "config_path", config_path);

    auto parser = std::make_shared<ov_core::YamlParser>(config_path);
    parser->set_node(node);
    std::string verbosity = "INFO";
    parser->parse_config("verbosity", verbosity);
    ov_core::Printer::setPrintLevel(verbosity);

    // Serial replay initializes synchronously, so a given bag always yields the same estimator call sequence.
    rcl_interfaces::msg::ParameterDescriptor fixed;
    fixed.read_only = true;
    for (const auto &name : {"multi_threading_subs", "multi_threading_pubs"})
      node->declare_parameter<bool>(name, false, fixed, true);
    VioManagerOptions params;
    params.print_and_load(parser);
    params.use_multi_threading_subs = false;
    params.use_multi_threading_pubs = false;
    const int num_cameras = params.state_options.num_cameras;

    const auto path_bag = ov_core::startup_parameter<std::string>(*node, "path_bag", "");
    const auto output_dir = ov_core::startup_parameter<std::string>(*node, "output_dir", "");
    const double bag_start = ov_core::startup_parameter<double>(*node, "bag_start", 0.0);
    const double bag_duration = ov_core::startup_parameter<double>(*node, "bag_duration", -1.0);
    const double sync_tolerance = ov_core::startup_parameter<double>(*node, "stereo_sync_tolerance", 0.005);
    const bool visualize = ov_core::startup_parameter<bool>(*node, "visualize", false);
    std::string topic_imu = "/imu0";
    parser->parse_external("relative_config_imu", "imu0", "rostopic", topic_imu);
    topic_imu = ov_core::startup_parameter<std::string>(*node, "topic_imu", topic_imu);
    std::map<std::string, int> camera_ids;
    for (int i = 0; i < num_cameras; i++) {
      std::string topic = "/cam" + std::to_string(i) + "/image_raw";
      parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "rostopic", topic);
      topic = ov_core::startup_parameter<std::string>(*node, "topic_camera" + std::to_string(i), topic);
      if (!camera_ids.emplace(topic, i).second || topic == topic_imu)
        throw std::invalid_argument("Camera and IMU topics must be distinct");
    }
    if (!parser->successful())
      throw std::invalid_argument("Unable to parse required OpenVINS parameters");
    if (path_bag.empty())
      throw std::invalid_argument("Set the path_bag parameter to a ROS 2 bag");
    if (num_cameras < 1 || !std::isfinite(params.track_frequency) || params.track_frequency <= 0 || !std::isfinite(bag_start) ||
        bag_start < 0 || !std::isfinite(bag_duration) || !std::isfinite(sync_tolerance) || sync_tolerance < 0)
      throw std::invalid_argument("Invalid camera count, tracking rate, bag window, or stereo tolerance");

    rosbag2_cpp::Reader reader;
    rosbag2_storage::StorageOptions storage;
    storage.uri = path_bag; // the storage plugin is detected from the bag metadata
    reader.open(storage);
    rosbag2_storage::StorageFilter filter;
    filter.topics.push_back(topic_imu);
    for (const auto &camera : camera_ids)
      filter.topics.push_back(camera.first);
    reader.set_filter(filter);
    std::map<std::string, std::string> topic_types;
    for (const auto &topic : reader.get_all_topics_and_types())
      topic_types[topic.name] = topic.type;
    auto require_topic = [&](const std::string &topic, std::initializer_list<const char *> types) {
      const auto found = topic_types.find(topic);
      if (found == topic_types.end()) {
        std::string available;
        for (const auto &item : topic_types)
          available += " " + item.first + " [" + item.second + "]";
        throw std::invalid_argument("Topic " + topic + " is not in the bag; available:" + available);
      }
      for (const char *type : types) {
        if (found->second == type)
          return;
      }
      throw std::invalid_argument("Topic " + topic + " has unsupported type " + found->second);
    };
    require_topic(topic_imu, {"sensor_msgs/msg/Imu"});
    for (const auto &camera : camera_ids)
      require_topic(camera.first, {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"});

    auto app = std::make_shared<VioManager>(params);
    std::shared_ptr<ROS2Visualizer> viz;
    if (visualize)
      viz = std::make_shared<ROS2Visualizer>(node, app, nullptr, parser);
    std::unique_ptr<TrajectoryWriter> trajectory;
    std::ofstream timing;
    if (!output_dir.empty()) {
      trajectory = std::make_unique<TrajectoryWriter>((boost::filesystem::path(output_dir) / "trajectory.txt").string());
      timing.open((boost::filesystem::path(output_dir) / "timing.csv").string(), std::ofstream::trunc);
      if (!timing)
        throw std::runtime_error("Unable to write timing.csv in " + output_dir);
      timing << "# camera timestamp(s),estimator seconds,initialized\n";
    }
    std::vector<double> processing;
    SerialFeeder feeder(app, [&](const ov_core::CameraData &packet, double seconds) {
      processing.push_back(seconds);
      if (trajectory) {
        trajectory->write(*app);
        char row[96];
        std::snprintf(row, sizeof(row), "%.9f,%.6f,%d\n", packet.timestamp, seconds, app->initialized() ? 1 : 0);
        timing << row;
      }
      if (viz)
        viz->visualize();
    });

    const auto masks = replay_masks(params);
    FrameSynchronizer synchronizer(num_cameras, sync_tolerance);
    double window_begin = std::numeric_limits<double>::quiet_NaN(), window_end = std::numeric_limits<double>::infinity();
    size_t messages = 0, decode_failures = 0, stale_images = 0;
    const auto wall_start = std::chrono::steady_clock::now();
    while (reader.has_next() && rclcpp::ok()) {
      const auto bag_message = reader.read_next();
      messages++;
      if (bag_message->topic_name == topic_imu) {
        const auto message = deserialize<sensor_msgs::msg::Imu>(*bag_message);
        ov_core::ImuData imu;
        imu.timestamp = stamp_seconds(message.header.stamp);
        if (std::isnan(window_begin)) {
          window_begin = imu.timestamp + bag_start;
          window_end = bag_duration < 0 ? window_end : window_begin + bag_duration;
        }
        if (imu.timestamp < window_begin)
          continue;
        if (imu.timestamp > window_end)
          break;
        imu.wm << message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z;
        imu.am << message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z;
        feeder.feed_imu(imu);
        continue;
      }

      // Camera message: raw or compressed, reduced to 8-bit grayscale like the live node
      const int camera = camera_ids.at(bag_message->topic_name);
      double timestamp = std::numeric_limits<double>::quiet_NaN();
      cv::Mat image;
      try {
        if (topic_types.at(bag_message->topic_name) == "sensor_msgs/msg/CompressedImage") {
          const auto compressed = deserialize<sensor_msgs::msg::CompressedImage>(*bag_message);
          timestamp = stamp_seconds(compressed.header.stamp);
          image = cv::imdecode(compressed.data, cv::IMREAD_GRAYSCALE);
        } else {
          const auto raw = deserialize<sensor_msgs::msg::Image>(*bag_message);
          timestamp = stamp_seconds(raw.header.stamp);
          image = cv_bridge::toCvCopy(raw, sensor_msgs::image_encodings::MONO8)->image;
        }
      } catch (const std::exception &error) {
        RCLCPP_WARN_ONCE(node->get_logger(), "Skipping undecodable image on %s: %s", bag_message->topic_name.c_str(), error.what());
      }
      if (image.empty()) {
        decode_failures++;
        continue;
      }
      if (std::isnan(window_begin) || timestamp < window_begin || timestamp > window_end)
        continue;
      if (image.size() != masks.at(camera).size())
        throw std::runtime_error("cam" + std::to_string(camera) + " image size does not match its calibration");
      if (!synchronizer.push(camera, timestamp, std::move(image))) {
        stale_images++;
        continue;
      }
      FrameSynchronizer::Frame frame;
      while (synchronizer.pop(frame)) {
        ov_core::CameraData packet;
        packet.timestamp = frame.timestamp;
        for (int id = 0; id < num_cameras; id++) {
          packet.sensor_ids.push_back(id);
          packet.images.push_back(frame.images.at(id));
          packet.masks.push_back(masks.at(id));
        }
        feeder.feed_camera(std::move(packet));
      }
    }
    feeder.finish();
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();

    const auto &stats = feeder.statistics();
    if (trajectory) {
      write_replay_summary((boost::filesystem::path(output_dir) / "summary.json").string(), path_bag, stats, *trajectory, processing, wall,
                           {{"bag_messages", static_cast<double>(messages)},
                            {"unmatched_images", static_cast<double>(synchronizer.unmatched())},
                            {"undecodable_images", static_cast<double>(decode_failures)},
                            {"stale_images", static_cast<double>(stale_images)}});
    }
    RCLCPP_INFO(node->get_logger(), "Processed %zu frames (%zu unprocessed, %zu unmatched images) from %zu bag messages in %.2f s",
                stats.cameras_processed, stats.cameras_unprocessed, synchronizer.unmatched(), messages, wall);
    if (trajectory)
      RCLCPP_INFO(node->get_logger(), "Wrote %zu poses, digest %s", trajectory->count(), trajectory->digest().c_str());
    if (viz)
      viz->visualize_final();
    if (trajectory && trajectory->count() == 0) {
      RCLCPP_ERROR(node->get_logger(), "The estimator never initialized; no trajectory was produced");
      result = EXIT_FAILURE;
    }
    // Release image transport publishers before their static plugin loaders are destroyed.
    viz.reset();
  } catch (const std::exception &error) {
    RCLCPP_ERROR(rclcpp::get_logger("open_vins"), "ros2_serial_msckf failed: %s", error.what());
    result = EXIT_FAILURE;
  }
  rclcpp::shutdown();
  return result;
}
