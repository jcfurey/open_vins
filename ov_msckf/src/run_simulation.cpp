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

#include <csignal>
#include <memory>

#include "core/VioManager.h"
#include "sim/Simulator.h"
#include "utils/colors.h"
#include "utils/dataset_reader.h"
#if ROS_AVAILABLE == 2
#include "utils/ros2_parameters.h"
#endif
#include "utils/print.h"
#include "utils/sensor_data.h"

#if ROS_AVAILABLE == 1
#include "ros/ROS1Visualizer.h"
#include <ros/ros.h>
#elif ROS_AVAILABLE == 2
#include "ros/ROS2Visualizer.h"
#include <rclcpp/rclcpp.hpp>
#endif

using namespace ov_msckf;

#if ROS_AVAILABLE == 1
// Define the function to be called when ctrl-c (SIGINT) is sent to process
void signal_callback_handler(int signum) { std::exit(signum); }
#endif

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
    ros::init(argc, argv, "run_simulation");
    auto nh = std::make_shared<ros::NodeHandle>("~");
    nh->param<std::string>("config_path", config_path, config_path);
#elif ROS_AVAILABLE == 2
    // Launch our ros node
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.allow_undeclared_parameters(false);
    options.automatically_declare_parameters_from_overrides(false);
    auto node = std::make_shared<rclcpp::Node>("run_simulation", options);
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
    std::string verbosity = "INFO";
    parser->parse_config("verbosity", verbosity);
    ov_core::Printer::setPrintLevel(verbosity);

    // Create our VIO system
#if ROS_AVAILABLE == 2
    rcl_interfaces::msg::ParameterDescriptor worker_setting;
    worker_setting.read_only = true;
    for (const auto &name : {"multi_threading_subs", "multi_threading_pubs"})
      node->declare_parameter<bool>(name, false, worker_setting, true);
    node->declare_parameter<int>("num_opencv_threads", 0, worker_setting, true);
#endif
    VioManagerOptions params;
    params.print_and_load(parser);
    params.print_and_load_simulation(parser);
    params.num_opencv_threads = 0; // for repeatability
    params.use_multi_threading_pubs = false;
    params.use_multi_threading_subs = false;
    if (!parser->successful())
      throw std::invalid_argument("Unable to parse required OpenVINS parameters");
    if (params.state_options.num_cameras < 1 || params.state_options.num_cameras > 64 || !std::isfinite(params.track_frequency) ||
        params.track_frequency <= 0 || !std::isfinite(params.init_options.init_window_time) || params.init_options.init_window_time <= 0 ||
        params.grid_x <= 0 || params.grid_y <= 0 || params.min_px_dist <= 0 || params.num_pts <= 0)
      throw std::invalid_argument("Invalid camera count, tracking rate, initialization window, or feature grid");
    if (!std::isfinite(params.sim_freq_imu) || params.sim_freq_imu <= 0 || !std::isfinite(params.sim_freq_cam) || params.sim_freq_cam <= 0)
      throw std::invalid_argument("Simulation rates must be finite and positive");
    auto sim = std::make_shared<Simulator>(params);
    auto sys = std::make_shared<VioManager>(params);
#if ROS_AVAILABLE == 1
    auto viz = std::make_shared<ROS1Visualizer>(nh, sys, sim);
#elif ROS_AVAILABLE == 2
    auto viz = std::make_shared<ROS2Visualizer>(node, sys, sim, parser);
#endif

    // Ensure we read in all parameters required
    if (!parser->successful()) {
      PRINT_ERROR(RED "unable to parse all parameters, please fix\n" RESET);
      throw std::invalid_argument("Unable to parse required OpenVINS parameters");
    }

    //===================================================================================
    //===================================================================================
    //===================================================================================

    // Get initial state
    // NOTE: we are getting it at the *next* timestep so we get the first IMU message
    double next_imu_time = sim->current_timestamp() + 1.0 / params.sim_freq_imu;
    Eigen::Matrix<double, 17, 1> imustate;
    bool success = sim->get_state(next_imu_time, imustate);
    if (!success) {
      PRINT_ERROR(RED "[SIM]: Could not initialize the filter to the first state\n" RESET);
      PRINT_ERROR(RED "[SIM]: Did the simulator load properly???\n" RESET);
      throw std::invalid_argument("Unable to parse required OpenVINS parameters");
    }

    // Since the state time is in the camera frame of reference
    // Subtract out the imu to camera time offset
    imustate(0, 0) -= sim->get_true_parameters().calib_camimu_dt;

    // Initialize our filter with the groundtruth
    sys->initialize_with_gt(imustate);

    //===================================================================================
    //===================================================================================
    //===================================================================================

    // Buffer our camera image
    double buffer_timecam = -1;
    std::vector<int> buffer_camids;
    std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> buffer_feats;

    // Step through the rosbag
#if ROS_AVAILABLE == 1
    while (sim->ok() && ros::ok()) {
#elif ROS_AVAILABLE == 2
    while (sim->ok() && rclcpp::ok()) {
#else
    while (sim->ok()) {
#endif

      // IMU: get the next simulated IMU measurement if we have it
      ov_core::ImuData message_imu;
      bool hasimu = sim->get_next_imu(message_imu.timestamp, message_imu.wm, message_imu.am);
      if (hasimu) {
        sys->feed_measurement_imu(message_imu);
#if ROS_AVAILABLE == 1 || ROS_AVAILABLE == 2
        viz->visualize_odometry(message_imu.timestamp);
#endif
      }

      // CAM: get the next simulated camera uv measurements if we have them
      double time_cam;
      std::vector<int> camids;
      std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> feats;
      bool hascam = sim->get_next_cam(time_cam, camids, feats);
      if (hascam) {
        if (buffer_timecam != -1) {
          sys->feed_measurement_simulation(buffer_timecam, buffer_camids, buffer_feats);
#if ROS_AVAILABLE == 1 || ROS_AVAILABLE == 2
          viz->visualize();
#endif
        }
        buffer_timecam = time_cam;
        buffer_camids = camids;
        buffer_feats = feats;
      }
    }

    // Final visualization
#if ROS_AVAILABLE == 1
    viz->visualize_final();
    ros::shutdown();
#elif ROS_AVAILABLE == 2
    viz->visualize_final();
    rclcpp::shutdown();
    // Release image transport publishers before their static plugin loaders are destroyed.
    viz.reset();
    sys.reset();
    sim.reset();
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
