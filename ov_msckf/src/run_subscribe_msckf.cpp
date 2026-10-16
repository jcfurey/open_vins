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

#include <memory>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "utils/dataset_reader.h"
#if ROS_AVAILABLE == 2
#include "utils/ros2_parameters.h"
#endif

#if ROS_AVAILABLE == 1
#include "ros/ROS1Visualizer.h"
#include <ros/ros.h>
#elif ROS_AVAILABLE == 2
#include "ros/ROS2Visualizer.h"
#include <rclcpp/rclcpp.hpp>
#endif

using namespace ov_msckf;

// Main function
int main(int argc, char **argv) {
  try {

    // Ensure we have a path, if the user passes it then we should use it
    std::string config_path = "unset_path_to_config.yaml";
    if (argc > 1) {
      config_path = argv[1];
    }

#if ROS_AVAILABLE == 1
    // Launch our ros node
    ros::init(argc, argv, "run_subscribe_msckf");
    auto nh = std::make_shared<ros::NodeHandle>("~");
    nh->param<std::string>("config_path", config_path, config_path);
#elif ROS_AVAILABLE == 2
    // Launch our ros node
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.allow_undeclared_parameters(false);
    options.automatically_declare_parameters_from_overrides(false);
    auto node = std::make_shared<rclcpp::Node>("run_subscribe_msckf", options);
    const auto positional = rclcpp::remove_ros_arguments(argc, argv);
    config_path = positional.size() > 1 ? positional[1] : "unset_path_to_config.yaml";
    config_path = ov_core::startup_parameter<std::string>(*node, "config_path", config_path);
#endif

    // Load the config
    auto parser = std::make_shared<ov_core::YamlParser>(config_path);
#if ROS_AVAILABLE == 1
    parser->set_node_handler(nh);
#elif ROS_AVAILABLE == 2
    parser->set_node(node);
#endif

    // Verbosity
    std::string verbosity = "DEBUG";
    parser->parse_config("verbosity", verbosity);
    ov_core::Printer::setPrintLevel(verbosity);

    // Create our VIO system
#if ROS_AVAILABLE == 2
    rcl_interfaces::msg::ParameterDescriptor worker_setting;
    worker_setting.read_only = true;
    for (const auto &name : {"multi_threading_subs", "multi_threading_pubs"})
      node->declare_parameter<bool>(name, false, worker_setting, true);
#endif
    VioManagerOptions params;
    params.print_and_load(parser);
#if ROS_AVAILABLE == 1
    params.use_multi_threading_subs = true;
#else
    // ROS callbacks only enqueue; the managed estimator worker owns initialization and state.
    params.use_multi_threading_subs = false;
    params.use_multi_threading_pubs = false;
#endif
    if (!parser->successful())
      throw std::invalid_argument("Unable to parse required OpenVINS parameters");
    if (params.state_options.num_cameras < 1 || params.state_options.num_cameras > 64 || !std::isfinite(params.track_frequency) ||
        params.track_frequency <= 0 || !std::isfinite(params.init_options.init_window_time) || params.init_options.init_window_time <= 0 ||
        params.grid_x <= 0 || params.grid_y <= 0 || params.min_px_dist <= 0 || params.num_pts <= 0)
      throw std::invalid_argument("Invalid camera count, tracking rate, initialization window, or feature grid");
    auto sys = std::make_shared<VioManager>(params);
#if ROS_AVAILABLE == 1
    auto viz = std::make_shared<ROS1Visualizer>(nh, sys);
    viz->setup_subscribers(parser);
#elif ROS_AVAILABLE == 2
    auto viz = std::make_shared<ROS2Visualizer>(node, sys, nullptr, parser);
    viz->setup_subscribers(parser);
    sys.reset(); // The visualizer owns the live estimator, including resets.
#endif

    // Ensure we read in all parameters required
    if (!parser->successful()) {
      PRINT_ERROR(RED "unable to parse all parameters, please fix\n" RESET);
      throw std::invalid_argument("Unable to parse required OpenVINS parameters");
    }

    // Spin off to ROS
    PRINT_DEBUG("done...spinning to ros\n");
#if ROS_AVAILABLE == 1
    // ros::spin();
    ros::AsyncSpinner spinner(0);
    spinner.start();
    ros::waitForShutdown();
#elif ROS_AVAILABLE == 2
    // rclcpp::spin(node);
    const int executor_threads = ov_core::startup_parameter<int>(*node, "executor_threads", 2);
    if (executor_threads < 1 || executor_threads > 16)
      throw std::invalid_argument("executor_threads must be between 1 and 16");
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions{}, executor_threads);
    executor.add_node(node);
    executor.spin();
#endif

    // Quiesce the worker before reading its state for final reporting.
#if ROS_AVAILABLE == 2
    viz->stop();
    if (!viz->healthy())
      throw std::runtime_error("Estimator worker failed");
#endif
    // Final visualization
    viz->visualize_final();
#if ROS_AVAILABLE == 1
    ros::shutdown();
#elif ROS_AVAILABLE == 2
    rclcpp::shutdown();
    // Release image transport publishers before their static plugin loaders are destroyed.
    viz.reset();
    sys.reset();
#endif

    // Done!
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
#if ROS_AVAILABLE == 2
    RCLCPP_ERROR(rclcpp::get_logger("open_vins"), "OpenVINS failed: %s", error.what());
    rclcpp::shutdown();
#else
    std::cerr << "OpenVINS failed: " << error.what() << std::endl;
#endif
    return EXIT_FAILURE;
  }
}
