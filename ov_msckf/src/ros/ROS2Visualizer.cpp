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

#include "ROS2Visualizer.h"
#include "Ros2ResetOptions.h"

#include <type_traits>

#include "core/VioManager.h"
#include "ros/ROSVisualizerHelper.h"
#include "ros/Ros2Compat.h"
#include "ros/Ros2Covariance.h"
#include "ros/Ros2SensorQos.h"
#include "sim/Simulator.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/dataset_reader.h"
#include "utils/print.h"
#include "utils/ros2_parameters.h"
#include "utils/sensor_data.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

namespace {
template <typename Broadcaster, typename = void> struct HasTfNodeInterfaces : std::false_type {};
template <typename Broadcaster>
struct HasTfNodeInterfaces<Broadcaster, std::void_t<typename Broadcaster::RequiredInterfaces>> : std::true_type {};

template <typename Broadcaster>
std::shared_ptr<Broadcaster> make_tf_broadcaster(const std::shared_ptr<rclcpp::Node> &node, const rclcpp::QoS &qos,
                                                const rclcpp::PublisherOptions &options) {
  if constexpr (HasTfNodeInterfaces<Broadcaster>::value) {
    return std::make_shared<Broadcaster>(
        typename Broadcaster::RequiredInterfaces(node->get_node_parameters_interface(), node->get_node_topics_interface()), qos, options);
  } else {
    return std::make_shared<Broadcaster>(node, qos, options);
  }
}

double sensor_time(const builtin_interfaces::msg::Time &stamp) {
  if (stamp.sec < 0 || stamp.nanosec >= 1000000000U)
    return std::numeric_limits<double>::quiet_NaN();
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

void forward_stereo(const std::weak_ptr<ROS2Visualizer> &visualizer, const sensor_msgs::msg::Image::ConstSharedPtr &left,
                    const sensor_msgs::msg::Image::ConstSharedPtr &right) {
  if (const auto self = visualizer.lock())
    self->callback_stereo(left, right, 0, 1);
}
} // namespace

ROS2Visualizer::ROS2Visualizer(std::shared_ptr<rclcpp::Node> node, std::shared_ptr<VioManager> app, std::shared_ptr<Simulator> sim,
                               std::shared_ptr<YamlParser> parser)
    : _node(std::move(node)), _app(std::move(app)), _sim(std::move(sim)),
      sensor_options(std::make_unique<VioManagerOptions>(copy_ros2_reset_options(_app->get_params()))) {

  node = _node;
  auto integer = [&](const std::string &name, int value, int minimum, int maximum) {
    if (parser)
      parser->parse_config(name, value, false);
    value = startup_parameter<int>(*node, name, value);
    if (value < minimum || value > maximum)
      throw std::invalid_argument(name + " is outside its supported range");
    return value;
  };
  auto real = [&](const std::string &name, double value, double minimum) {
    if (parser)
      parser->parse_config(name, value, false);
    value = startup_parameter<double>(*node, name, value);
    if (!std::isfinite(value) || value < minimum)
      throw std::invalid_argument(name + " must be finite and >= " + std::to_string(minimum));
    return value;
  };
  auto boolean = [&](const std::string &name, bool value) {
    if (parser)
      parser->parse_config(name, value, false);
    return startup_parameter<bool>(*node, name, value);
  };
  auto text = [&](const std::string &name, std::string value) {
    if (parser)
      parser->parse_config(name, value, false);
    return startup_parameter<std::string>(*node, name, value);
  };
  const int max_imu_queue = integer("max_imu_queue_size", 4096, 2, 1000000);
  const int max_camera_queue = integer("max_camera_queue_size", 20, 1, 10000);
  imu_qos_depth = integer("imu_qos_depth", 200, 1, 100000);
  camera_qos_depth = integer("camera_qos_depth", 10, 1, 1000);
  path_history_size = integer("path_history_size", 2000, 0, 1000000);
  stereo_sync_tolerance = real("stereo_sync_tolerance", 0.005, 0.0);
  odometry_publish_rate = real("odometry_publish_rate", 200.0, 0.0);
  track_image_publish_rate = real("track_image_publish_rate", 10.0, 0.0);
  max_imu_history_seconds = real("max_imu_history_seconds", 10.0, sensor_options->init_options.init_window_time + 0.2);
  global_frame = text("global_frame_id", global_frame);
  imu_frame = text("imu_frame_id", imu_frame);
  camera_frame_prefix = text("camera_frame_prefix", camera_frame_prefix);
  truth_frame = text("truth_frame_id", truth_frame);
  for (const auto &frame : {global_frame, imu_frame, camera_frame_prefix, truth_frame}) {
    if (frame.empty() || frame.front() == '/' || frame.find_first_of(" \t\n") != std::string::npos)
      throw std::invalid_argument("TF frame names must be nonempty relative names without whitespace");
  }
  if (global_frame == imu_frame || global_frame == truth_frame || imu_frame == truth_frame)
    throw std::invalid_argument("Global, IMU, and truth frames must be distinct");
  for (int id = 0; id < sensor_options->state_options.num_cameras; ++id) {
    const auto frame = camera_frame_prefix + std::to_string(id);
    if (frame == global_frame || frame == imu_frame || frame == truth_frame)
      throw std::invalid_argument("Camera TF frames must be distinct from global, IMU, and truth frames");
  }
  const bool reset_on_jump = boolean("reset_on_time_jump", true);
  const double max_camera_gap = real("max_camera_gap_seconds", 1.0, 0.001);
  measurements = std::make_unique<Ros2MeasurementBuffer>(max_imu_queue, max_camera_queue, sensor_options->state_options.num_cameras,
                                                         sensor_options->track_frequency, reset_on_jump, max_camera_gap);
  input_masks = sensor_options->input_masks();
  for (const auto &mask : input_masks)
    image_sizes.emplace(mask.first, mask.second.size());

  // Setup our transform broadcaster
  rclcpp::PublisherOptions tf_options;
  tf_options.qos_overriding_options = rclcpp::QosOverridingOptions{
      rclcpp::QosPolicyKind::Depth, rclcpp::QosPolicyKind::Durability, rclcpp::QosPolicyKind::History, rclcpp::QosPolicyKind::Reliability};
  // Capture the status storage independently of this object's lifetime.
  const auto tf_interest = tf_subscribers;
  tf_options.event_callbacks.matched_callback = [tf_interest](rclcpp::MatchedInfo &status) {
    tf_interest->store(status.current_count > 0, std::memory_order_relaxed);
  };
  mTfBr = make_tf_broadcaster<tf2_ros::TransformBroadcaster>(node, tf2_ros::DynamicBroadcasterQoS(), tf_options);
  rclcpp::PublisherOptions static_tf_options;
  static_tf_options.qos_overriding_options = rclcpp::QosOverridingOptions{
      rclcpp::QosPolicyKind::Depth, rclcpp::QosPolicyKind::History, rclcpp::QosPolicyKind::Reliability};
  static_tf_broadcaster = make_tf_broadcaster<tf2_ros::StaticTransformBroadcaster>(node, tf2_ros::StaticBroadcasterQoS(), static_tf_options);

  // Setup pose and path publisher
  pub_poseimu = node->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("poseimu", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_poseimu->get_topic_name());
  pub_odomimu = node->create_publisher<nav_msgs::msg::Odometry>("odomimu", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_odomimu->get_topic_name());
  pub_pathimu = node->create_publisher<nav_msgs::msg::Path>("pathimu", rclcpp::QoS(1).transient_local());
  PRINT_DEBUG("Publishing: %s\n", pub_pathimu->get_topic_name());

  // 3D points publishing
  pub_points_msckf = node->create_publisher<sensor_msgs::msg::PointCloud2>("points_msckf", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_points_msckf->get_topic_name());
  pub_points_slam = node->create_publisher<sensor_msgs::msg::PointCloud2>("points_slam", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_points_msckf->get_topic_name());
  pub_points_aruco = node->create_publisher<sensor_msgs::msg::PointCloud2>("points_aruco", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_points_aruco->get_topic_name());
  pub_points_sim = node->create_publisher<sensor_msgs::msg::PointCloud2>("points_sim", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_points_sim->get_topic_name());

  // Our tracking image
  it_pub_tracks = create_image_publisher(*node, "trackhist", rclcpp::QoS(2));
  PRINT_DEBUG("Publishing: %s\n", it_pub_tracks.getTopic().c_str());

  // Groundtruth publishers
  pub_posegt = node->create_publisher<geometry_msgs::msg::PoseStamped>("posegt", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_posegt->get_topic_name());
  pub_pathgt = node->create_publisher<nav_msgs::msg::Path>("pathgt", rclcpp::QoS(1).transient_local());
  PRINT_DEBUG("Publishing: %s\n", pub_pathgt->get_topic_name());

  // Loop closure publishers
  pub_loop_pose = node->create_publisher<nav_msgs::msg::Odometry>("loop_pose", 2);
  pub_loop_point = node->create_publisher<sensor_msgs::msg::PointCloud>("loop_feats", 2);
  pub_loop_extrinsic = node->create_publisher<nav_msgs::msg::Odometry>("loop_extrinsic", 2);
  pub_loop_intrinsics = node->create_publisher<sensor_msgs::msg::CameraInfo>("loop_intrinsics", 2);
  it_pub_loop_img_depth = create_image_publisher(*node, "loop_depth", rclcpp::QoS(2));
  it_pub_loop_img_depth_color = create_image_publisher(*node, "loop_depth_colored", rclcpp::QoS(2));

  publish_global2imu_tf = boolean("publish_global_to_imu_tf", true);
  publish_calibration_tf = boolean("publish_calibration_tf", true);
  const std::string path_to_gt = text("path_gt", "");
  if (_sim == nullptr && !path_to_gt.empty())
    DatasetReader::load_gt_file(path_to_gt, gt_states);
  save_total_state = boolean("save_total_state", false);
  const std::string filepath_est = text("filepath_est", "state_estimate.txt");
  const std::string filepath_std = text("filepath_std", "state_deviation.txt");
  const std::string filepath_gt = text("filepath_gt", "state_groundtruth.txt");
  if (save_total_state) {

    // If it exists, then delete it
    if (boost::filesystem::exists(filepath_est))
      boost::filesystem::remove(filepath_est);
    if (boost::filesystem::exists(filepath_std))
      boost::filesystem::remove(filepath_std);

    // Create folder path to this location if not exists
    for (const auto &path : {filepath_est, filepath_std}) {
      const auto parent = boost::filesystem::path(path).parent_path();
      if (!parent.empty())
        boost::filesystem::create_directories(parent);
    }

    // Open the files
    of_state_est.open(filepath_est.c_str());
    of_state_std.open(filepath_std.c_str());
    if (!of_state_est || !of_state_std)
      throw std::runtime_error("Unable to open estimator state output files");
    of_state_est << "# timestamp(s) q p v bg ba cam_imu_dt num_cam cam0_k cam0_d cam0_rot cam0_trans ... imu_model dw da tg wtoI atoI etc"
                 << std::endl;
    of_state_std << "# timestamp(s) q p v bg ba cam_imu_dt num_cam cam0_k cam0_d cam0_rot cam0_trans ... imu_model dw da tg wtoI atoI etc"
                 << std::endl;

    // Groundtruth if we are simulating
    if (_sim != nullptr) {
      if (boost::filesystem::exists(filepath_gt))
        boost::filesystem::remove(filepath_gt);
      const auto parent = boost::filesystem::path(filepath_gt).parent_path();
      if (!parent.empty())
        boost::filesystem::create_directories(parent);
      of_state_gt.open(filepath_gt.c_str());
      if (!of_state_gt)
        throw std::runtime_error("Unable to open simulation ground-truth output");
      of_state_gt << "# timestamp(s) q p v bg ba cam_imu_dt num_cam cam0_k cam0_d cam0_rot cam0_trans ... imu_model dw da tg wtoI atoI etc"
                  << std::endl;
    }
  }
}

ROS2Visualizer::~ROS2Visualizer() { stop(); }

void ROS2Visualizer::stop() {
  if (stopped.exchange(true))
    return;
  measurements->stop();
  if (estimator_thread.joinable())
    estimator_thread.join();
  const auto stats = measurements->statistics();
  RCLCPP_INFO(_node->get_logger(),
              "ROS2 ingress: imu=%lu cameras=%lu dropped_imu=%lu dropped_cameras=%lu invalid=%lu out_of_order=%lu "
              "rate_limited=%lu peak_imu=%zu peak_cameras=%zu resets=%lu",
              stats.imu_accepted, stats.cameras_accepted, stats.imu_dropped, stats.cameras_dropped, stats.invalid, stats.out_of_order,
              stats.rate_limited, stats.imu_peak, stats.camera_peak, stats.resets);
  RCLCPP_INFO(_node->get_logger(), "ROS2 outputs: fast_odometry=%zu tf_subscribers=%d", fast_odometry_updates,
              static_cast<int>(tf_subscribers->load(std::memory_order_relaxed)));
}

void ROS2Visualizer::reset_estimator() {
  RCLCPP_WARN(_node->get_logger(), "Resetting estimator after sensor loss, a time discontinuity, or exceeded history limit");
  auto options = copy_ros2_reset_options(*sensor_options);
  options.use_multi_threading_subs = false;
  options.use_multi_threading_pubs = false;
  _app = std::make_shared<VioManager>(options);
  poses_imu.clear();
  poses_gt.clear();
  last_camera_time = last_odometry_time = -1.0;
  static_calibration_published = false;
  last_visualization_timestamp = last_visualization_timestamp_image = -1.0;
  start_time_set = false;
}

void ROS2Visualizer::run_estimator() {
  double fed_imu_time = -1.0;
  double last_camera_imu_time = -1.0;
  try {
    Ros2MeasurementBuffer::Batch batch;
    while (measurements->wait_take(batch, fed_imu_time, _app->get_state()->_calib_dt_CAMtoIMU->value()(0))) {
      if (batch.reset || (!batch.imu.empty() && _app->get_state()->_timestamp >= 0 && last_camera_imu_time >= 0 &&
                          batch.imu.back().timestamp - last_camera_imu_time > max_imu_history_seconds)) {
        reset_estimator();
        fed_imu_time = last_camera_imu_time = -1.0;
      }
      for (const auto &imu : batch.imu) {
        _app->feed_measurement_imu(imu);
        fed_imu_time = imu.timestamp;
      }
      if (batch.camera) {
        last_camera_time = batch.camera->timestamp;
        last_camera_imu_time = last_camera_time + _app->get_state()->_calib_dt_CAMtoIMU->value()(0);
        const bool loopclosure_requested = has_loopclosure_subscribers();
        _app->set_active_tracks_enabled(loopclosure_requested);
        _app->set_active_image_enabled(loopclosure_requested);
        _app->feed_measurement_camera(*batch.camera);
        visualize();
      }
      if (fed_imu_time >= 0 && !stopped)
        visualize_odometry(fed_imu_time);
    }
  } catch (const std::exception &error) {
    if (stopped || !_node->get_node_base_interface()->get_context()->is_valid())
      return; // Context shutdown can interrupt an in-flight publication during SIGINT.
    worker_failed = true;
    measurements->stop();
    RCLCPP_ERROR(_node->get_logger(), "Estimator worker failed: %s", error.what());
    _node->get_node_base_interface()->get_context()->shutdown("OpenVINS estimator worker failed");
  }
}

void ROS2Visualizer::report_ingress(Ros2MeasurementBuffer::Result result, const char *sensor) {
  using Result = Ros2MeasurementBuffer::Result;
  if (result == Result::Invalid || result == Result::OutOfOrder || result == Result::DroppedOldest || result == Result::Reset) {
    RCLCPP_WARN_THROTTLE(_node->get_logger(), *_node->get_clock(), 2000,
                         "%s ingress result %d (invalid, stale, or overloaded input); inspect shutdown ingress counters", sensor,
                         static_cast<int>(result));
  }
}

builtin_interfaces::msg::Time ROS2Visualizer::measurement_stamp() const {
  const auto state = _app->get_state();
  const double camera_time = state->_timestamp >= 0 ? state->_timestamp : last_camera_time;
  return ROSVisualizerHelper::get_time_from_seconds(camera_time + state->_calib_dt_CAMtoIMU->value()(0));
}

void ROS2Visualizer::setup_subscribers(std::shared_ptr<YamlParser> parser) {
  if (!parser || !parser->successful())
    throw std::invalid_argument("Invalid OpenVINS configuration");
  if (_app->get_params().use_multi_threading_subs)
    throw std::invalid_argument("ROS2 requires synchronous initialization inside its estimator worker");
  const auto weak_self = weak_from_this();
  if (weak_self.expired())
    throw std::logic_error("ROS2Visualizer subscriptions require shared ownership");
  imu_callback_group = _node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  camera_callback_group = _node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  rclcpp::SubscriptionOptions imu_subscription_options, camera_subscription_options;
  imu_subscription_options.callback_group = imu_callback_group;
  camera_subscription_options.callback_group = camera_callback_group;
  imu_subscription_options.qos_overriding_options = sensor_qos_overrides(100000);
  camera_subscription_options.qos_overriding_options = sensor_qos_overrides(1000);
  auto imu_qos = rclcpp::SensorDataQoS().keep_last(imu_qos_depth);
  auto camera_qos = rclcpp::SensorDataQoS().keep_last(camera_qos_depth);

  std::string imu_topic = "/imu0";
  parser->parse_external("relative_config_imu", "imu0", "rostopic", imu_topic);
  imu_topic = startup_parameter<std::string>(*_node, "topic_imu", imu_topic);
  sub_imu = _node->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic, imu_qos,
      [weak_self](sensor_msgs::msg::Imu::ConstSharedPtr message) {
        if (const auto self = weak_self.lock())
          self->callback_inertial(message);
      },
      imu_subscription_options);
  RCLCPP_INFO(_node->get_logger(), "subscribing to IMU: %s", imu_topic.c_str());

  std::vector<std::string> camera_topics;
  for (int i = 0; i < sensor_options->state_options.num_cameras; ++i) {
    std::string topic = "/cam" + std::to_string(i) + "/image_raw";
    parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "rostopic", topic);
    camera_topics.push_back(startup_parameter<std::string>(*_node, "topic_camera" + std::to_string(i), topic));
  }
  if (!parser->successful())
    throw std::invalid_argument("Camera/IMU topics could not be loaded");
  if (sensor_options->state_options.num_cameras == 2) {
    using ImageFilter = message_filters::Subscriber<sensor_msgs::msg::Image>;
    auto left = create_filter_subscriber<ImageFilter>(_node, camera_topics[0], camera_qos, camera_subscription_options);
    auto right = create_filter_subscriber<ImageFilter>(_node, camera_topics[1], camera_qos, camera_subscription_options);
    sync_pol policy(camera_qos_depth);
    policy.setMaxIntervalDuration(rclcpp::Duration::from_seconds(stereo_sync_tolerance));
    auto sync = std::make_shared<message_filters::Synchronizer<sync_pol>>(std::move(policy), *left, *right);
    // A bind expression accepts both the exact image pair (Lyrical) and Jazzy's trailing null-message arguments.
    sync->registerCallback(std::bind(&forward_stereo, weak_self, std::placeholders::_1, std::placeholders::_2));
    sync_subs_cam.push_back(left);
    sync_subs_cam.push_back(right);
    sync_cam.push_back(sync);
    for (const auto &topic : camera_topics)
      RCLCPP_INFO(_node->get_logger(), "subscribing to cam (stereo): %s", topic.c_str());
  } else {
    for (size_t i = 0; i < camera_topics.size(); ++i) {
      subs_cam.push_back(_node->create_subscription<sensor_msgs::msg::Image>(
          camera_topics[i], camera_qos,
          [weak_self, i](sensor_msgs::msg::Image::ConstSharedPtr image) {
            if (const auto self = weak_self.lock())
              self->callback_monocular(image, i);
          },
          camera_subscription_options));
      RCLCPP_INFO(_node->get_logger(), "subscribing to cam (mono): %s", camera_topics[i].c_str());
    }
  }
  estimator_thread = std::thread(&ROS2Visualizer::run_estimator, this);
}

void ROS2Visualizer::visualize() {

  // Return if we have already visualized
  if (last_visualization_timestamp == _app->get_state()->_timestamp && _app->initialized())
    return;
  last_visualization_timestamp = _app->get_state()->_timestamp;

  // Start timing
  // boost::posix_time::ptime rT0_1, rT0_2;
  // rT0_1 = boost::posix_time::microsec_clock::local_time();

  // Visualization and all estimator reads share the estimator worker.
  publish_images();

  // Return if we have not inited
  if (!_app->initialized())
    return;

  // Save the start time of this dataset
  if (!start_time_set) {
    rT1 = boost::posix_time::microsec_clock::local_time();
    start_time_set = true;
  }

  // publish state
  publish_state();

  // publish points
  publish_features();

  // Publish gt if we have it
  publish_groundtruth();

  // Publish keyframe information
  publish_loopclosure_information();

  // Save total state
  if (save_total_state) {
    ROSVisualizerHelper::sim_save_total_state_to_file(_app->get_state(), _sim, of_state_est, of_state_std, of_state_gt);
  }

  // Print how much time it took to publish / displaying things
  // rT0_2 = boost::posix_time::microsec_clock::local_time();
  // double time_total = (rT0_2 - rT0_1).total_microseconds() * 1e-6;
  // PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for visualization\n" RESET, time_total);
}

void ROS2Visualizer::visualize_odometry(double timestamp) {

  // Return if we have not inited and a second has passes
  if (!_app->initialized() || (timestamp - _app->initialized_time()) < 1)
    return;

  if (odometry_publish_rate == 0)
    return;
  const auto state = _app->get_state();
  const bool dynamic_calibration = state->_options.do_calib_camera_pose;
  const bool tf_requested = tf_subscribers->load(std::memory_order_relaxed);
  const bool odometry_requested = pub_odomimu->get_subscription_count() != 0;
  const bool global_tf_requested = publish_global2imu_tf && tf_requested;
  const bool dynamic_calibration_requested = publish_calibration_tf && dynamic_calibration && tf_requested;

  auto publish_camera_calibration = [&](bool dynamic) {
    std::vector<geometry_msgs::msg::TransformStamped> transforms;
    transforms.reserve(state->_calib_IMUtoCAM.size());
    for (const auto &calib : state->_calib_IMUtoCAM) {
      auto trans_calib = ROSVisualizerHelper::get_stamped_transform_from_pose(_node, calib.second, true);
      trans_calib.header.stamp = ROSVisualizerHelper::get_time_from_seconds(timestamp);
      trans_calib.header.frame_id = imu_frame;
      trans_calib.child_frame_id = camera_frame_prefix + std::to_string(calib.first);
      transforms.push_back(std::move(trans_calib));
    }
    if (dynamic)
      mTfBr->sendTransform(transforms);
    else
      static_tf_broadcaster->sendTransform(transforms);
  };
  // Fixed calibration needs no state prediction, and must remain available to late listeners.
  if (publish_calibration_tf && !dynamic_calibration && !static_calibration_published) {
    publish_camera_calibration(false);
    static_calibration_published = true;
  }
  if (!odometry_requested && !global_tf_requested && !dynamic_calibration_requested)
    return;
  if (last_odometry_time >= 0 && timestamp - last_odometry_time < 1.0 / odometry_publish_rate - 1e-6)
    return;

  // Get fast propagate state at the desired timestamp
  Eigen::Matrix<double, 13, 1> state_plus = Eigen::Matrix<double, 13, 1>::Zero();
  Eigen::Matrix<double, 12, 12> cov_plus = Eigen::Matrix<double, 12, 12>::Zero();
  if (!_app->get_propagator()->fast_state_propagate(state, timestamp, state_plus, cov_plus))
    return;
  ++fast_odometry_updates;

  last_odometry_time = timestamp;

  // Publish our odometry message if requested
  if (odometry_requested) {

    // Our odometry message
    nav_msgs::msg::Odometry odomIinM;
    odomIinM.header.stamp = ROSVisualizerHelper::get_time_from_seconds(timestamp);
    odomIinM.header.frame_id = global_frame;

    // The POSE component (orientation and position)
    odomIinM.pose.pose.orientation.x = state_plus(0);
    odomIinM.pose.pose.orientation.y = state_plus(1);
    odomIinM.pose.pose.orientation.z = state_plus(2);
    odomIinM.pose.pose.orientation.w = state_plus(3);
    odomIinM.pose.pose.position.x = state_plus(4);
    odomIinM.pose.pose.position.y = state_plus(5);
    odomIinM.pose.pose.position.z = state_plus(6);

    // The TWIST component (angular and linear velocities)
    odomIinM.child_frame_id = imu_frame;
    odomIinM.twist.twist.linear.x = state_plus(7);   // vel in local frame
    odomIinM.twist.twist.linear.y = state_plus(8);   // vel in local frame
    odomIinM.twist.twist.linear.z = state_plus(9);   // vel in local frame
    odomIinM.twist.twist.angular.x = state_plus(10); // we do not estimate this...
    odomIinM.twist.twist.angular.y = state_plus(11); // we do not estimate this...
    odomIinM.twist.twist.angular.z = state_plus(12); // we do not estimate this...

    // Finally set the covariance in the message (in the order position then orientation as per ros convention)
    Eigen::Matrix<double, 12, 12> Phi = Eigen::Matrix<double, 12, 12>::Zero();
    Phi.block(0, 3, 3, 3).setIdentity();
    Phi.block(3, 0, 3, 3) = quat_2_Rot(state_plus.head<4>()).transpose();
    Phi.block(6, 6, 6, 6).setIdentity();
    cov_plus = Phi * cov_plus * Phi.transpose();
    for (int r = 0; r < 6; r++) {
      for (int c = 0; c < 6; c++) {
        odomIinM.pose.covariance[6 * r + c] = cov_plus(r, c);
      }
    }
    for (int r = 0; r < 6; r++) {
      for (int c = 0; c < 6; c++) {
        odomIinM.twist.covariance[6 * r + c] = cov_plus(r + 6, c + 6);
      }
    }
    pub_odomimu->publish(odomIinM);
  }

  // Publish our transform on TF
  // NOTE: since we use JPL we have an implicit conversion to Hamilton when we publish
  // NOTE: a rotation from GtoI in JPL has the same xyzw as a ItoG Hamilton rotation
  if (global_tf_requested) {
    auto odom_pose = std::make_shared<ov_type::PoseJPL>();
    odom_pose->set_value(state_plus.block(0, 0, 7, 1));
    geometry_msgs::msg::TransformStamped trans = ROSVisualizerHelper::get_stamped_transform_from_pose(_node, odom_pose, false);
    trans.header.stamp = ROSVisualizerHelper::get_time_from_seconds(timestamp);
    trans.header.frame_id = global_frame;
    trans.child_frame_id = imu_frame;
    mTfBr->sendTransform(trans);
  }

  if (dynamic_calibration_requested)
    publish_camera_calibration(true);
}

void ROS2Visualizer::visualize_final() {

  // Final time offset value
  if (_app->get_state()->_options.do_calib_camera_timeoffset) {
    PRINT_INFO(REDPURPLE "camera-imu timeoffset = %.5f\n\n" RESET, _app->get_state()->_calib_dt_CAMtoIMU->value()(0));
  }

  // Final camera intrinsics
  if (_app->get_state()->_options.do_calib_camera_intrinsics) {
    for (int i = 0; i < _app->get_state()->_options.num_cameras; i++) {
      std::shared_ptr<Vec> calib = _app->get_state()->_cam_intrinsics.at(i);
      PRINT_INFO(REDPURPLE "cam%d intrinsics:\n" RESET, (int)i);
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f\n" RESET, calib->value()(0), calib->value()(1), calib->value()(2), calib->value()(3));
      PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,%.5f\n\n" RESET, calib->value()(4), calib->value()(5), calib->value()(6), calib->value()(7));
    }
  }

  // Final camera extrinsics
  if (_app->get_state()->_options.do_calib_camera_pose) {
    for (int i = 0; i < _app->get_state()->_options.num_cameras; i++) {
      std::shared_ptr<PoseJPL> calib = _app->get_state()->_calib_IMUtoCAM.at(i);
      Eigen::Matrix4d T_CtoI = Eigen::Matrix4d::Identity();
      T_CtoI.block(0, 0, 3, 3) = quat_2_Rot(calib->quat()).transpose();
      T_CtoI.block(0, 3, 3, 1) = -T_CtoI.block(0, 0, 3, 3) * calib->pos();
      PRINT_INFO(REDPURPLE "T_C%dtoI:\n" RESET, i);
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f,\n" RESET, T_CtoI(0, 0), T_CtoI(0, 1), T_CtoI(0, 2), T_CtoI(0, 3));
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f,\n" RESET, T_CtoI(1, 0), T_CtoI(1, 1), T_CtoI(1, 2), T_CtoI(1, 3));
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f,\n" RESET, T_CtoI(2, 0), T_CtoI(2, 1), T_CtoI(2, 2), T_CtoI(2, 3));
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f\n\n" RESET, T_CtoI(3, 0), T_CtoI(3, 1), T_CtoI(3, 2), T_CtoI(3, 3));
    }
  }

  // IMU intrinsics
  if (_app->get_state()->_options.do_calib_imu_intrinsics) {
    Eigen::Matrix3d Dw = State::Dm(_app->get_state()->_options.imu_model, _app->get_state()->_calib_imu_dw->value());
    Eigen::Matrix3d Da = State::Dm(_app->get_state()->_options.imu_model, _app->get_state()->_calib_imu_da->value());
    Eigen::Matrix3d Tw = Dw.colPivHouseholderQr().solve(Eigen::Matrix3d::Identity());
    Eigen::Matrix3d Ta = Da.colPivHouseholderQr().solve(Eigen::Matrix3d::Identity());
    Eigen::Matrix3d R_IMUtoACC = _app->get_state()->_calib_imu_ACCtoIMU->Rot().transpose();
    Eigen::Matrix3d R_IMUtoGYRO = _app->get_state()->_calib_imu_GYROtoIMU->Rot().transpose();
    PRINT_INFO(REDPURPLE "Tw:\n" RESET);
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, Tw(0, 0), Tw(0, 1), Tw(0, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, Tw(1, 0), Tw(1, 1), Tw(1, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f\n\n" RESET, Tw(2, 0), Tw(2, 1), Tw(2, 2));
    PRINT_INFO(REDPURPLE "Ta:\n" RESET);
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, Ta(0, 0), Ta(0, 1), Ta(0, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, Ta(1, 0), Ta(1, 1), Ta(1, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f\n\n" RESET, Ta(2, 0), Ta(2, 1), Ta(2, 2));
    PRINT_INFO(REDPURPLE "R_IMUtoACC:\n" RESET);
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, R_IMUtoACC(0, 0), R_IMUtoACC(0, 1), R_IMUtoACC(0, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, R_IMUtoACC(1, 0), R_IMUtoACC(1, 1), R_IMUtoACC(1, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f\n\n" RESET, R_IMUtoACC(2, 0), R_IMUtoACC(2, 1), R_IMUtoACC(2, 2));
    PRINT_INFO(REDPURPLE "R_IMUtoGYRO:\n" RESET);
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, R_IMUtoGYRO(0, 0), R_IMUtoGYRO(0, 1), R_IMUtoGYRO(0, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, R_IMUtoGYRO(1, 0), R_IMUtoGYRO(1, 1), R_IMUtoGYRO(1, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f\n\n" RESET, R_IMUtoGYRO(2, 0), R_IMUtoGYRO(2, 1), R_IMUtoGYRO(2, 2));
  }

  // IMU intrinsics gravity sensitivity
  if (_app->get_state()->_options.do_calib_imu_g_sensitivity) {
    Eigen::Matrix3d Tg = State::Tg(_app->get_state()->_calib_imu_tg->value());
    PRINT_INFO(REDPURPLE "Tg:\n" RESET);
    PRINT_INFO(REDPURPLE "%.6f,%.6f,%.6f,\n" RESET, Tg(0, 0), Tg(0, 1), Tg(0, 2));
    PRINT_INFO(REDPURPLE "%.6f,%.6f,%.6f,\n" RESET, Tg(1, 0), Tg(1, 1), Tg(1, 2));
    PRINT_INFO(REDPURPLE "%.6f,%.6f,%.6f\n\n" RESET, Tg(2, 0), Tg(2, 1), Tg(2, 2));
  }

  // Publish RMSE if we have it
  if (!gt_states.empty()) {
    PRINT_INFO(REDPURPLE "RMSE: %.3f (deg) orientation\n" RESET, std::sqrt(summed_mse_ori / summed_number));
    PRINT_INFO(REDPURPLE "RMSE: %.3f (m) position\n\n" RESET, std::sqrt(summed_mse_pos / summed_number));
  }

  // Publish RMSE and NEES if doing simulation
  if (_sim != nullptr) {
    PRINT_INFO(REDPURPLE "RMSE: %.3f (deg) orientation\n" RESET, std::sqrt(summed_mse_ori / summed_number));
    PRINT_INFO(REDPURPLE "RMSE: %.3f (m) position\n\n" RESET, std::sqrt(summed_mse_pos / summed_number));
    PRINT_INFO(REDPURPLE "NEES: %.3f (deg) orientation\n" RESET, summed_nees_ori / summed_number);
    PRINT_INFO(REDPURPLE "NEES: %.3f (m) position\n\n" RESET, summed_nees_pos / summed_number);
  }

  // Print the total time
  rT2 = boost::posix_time::microsec_clock::local_time();
  PRINT_INFO(REDPURPLE "TIME: %.3f seconds\n\n" RESET, start_time_set ? (rT2 - rT1).total_microseconds() * 1e-6 : 0.0);
}

void ROS2Visualizer::callback_inertial(const sensor_msgs::msg::Imu::ConstSharedPtr msg) {
  ImuData imu;
  imu.timestamp = sensor_time(msg->header.stamp);
  imu.wm << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
  imu.am << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
  report_ingress(measurements->push_imu(imu), "IMU");
}

bool ROS2Visualizer::append_image(CameraData &message, const sensor_msgs::msg::Image::ConstSharedPtr &image, int camera_id) {
  try {
    const auto converted = cv_bridge::toCvShare(image, sensor_msgs::image_encodings::MONO8);
    if (converted->image.size() != image_sizes.at(camera_id)) {
      RCLCPP_WARN_THROTTLE(_node->get_logger(), *_node->get_clock(), 2000, "Camera %d image dimensions do not match its calibration",
                           camera_id);
      return false;
    }
    message.sensor_ids.push_back(camera_id);
    // The tracker retains and preprocesses these pixels; ROS message storage cannot outlive this callback.
    message.images.push_back(converted->image.clone());
    message.masks.push_back(input_masks.at(camera_id));
    return true;
  } catch (const std::exception &error) {
    RCLCPP_WARN_THROTTLE(_node->get_logger(), *_node->get_clock(), 2000, "Rejected camera image: %s", error.what());
    return false;
  }
}

void ROS2Visualizer::callback_monocular(const sensor_msgs::msg::Image::ConstSharedPtr image, int camera_id) {
  CameraData message;
  message.timestamp = sensor_time(image->header.stamp);
  if (append_image(message, image, camera_id))
    report_ingress(measurements->push_camera(std::move(message)), "camera");
}

void ROS2Visualizer::callback_stereo(const sensor_msgs::msg::Image::ConstSharedPtr left,
                                     const sensor_msgs::msg::Image::ConstSharedPtr right, int left_id, int right_id) {
  CameraData message;
  message.timestamp = sensor_time(left->header.stamp);
  const double right_time = sensor_time(right->header.stamp);
  if (!std::isfinite(message.timestamp) || !std::isfinite(right_time) ||
      std::abs(message.timestamp - right_time) > stereo_sync_tolerance + 1e-9)
    return;
  if (append_image(message, left, left_id) && append_image(message, right, right_id))
    report_ingress(measurements->push_camera(std::move(message)), "stereo camera");
}

void ROS2Visualizer::publish_state() {

  // Get the current state
  std::shared_ptr<State> state = _app->get_state();

  // We want to publish in the IMU clock frame
  // The timestamp in the state will be the last camera time
  double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
  double timestamp_inI = state->_timestamp + t_ItoC;

  // Create pose of IMU (note we use the bag time)
  geometry_msgs::msg::PoseWithCovarianceStamped poseIinM;
  poseIinM.header.stamp = ROSVisualizerHelper::get_time_from_seconds(timestamp_inI);
  poseIinM.header.frame_id = global_frame;
  poseIinM.pose.pose.orientation.x = state->_imu->quat()(0);
  poseIinM.pose.pose.orientation.y = state->_imu->quat()(1);
  poseIinM.pose.pose.orientation.z = state->_imu->quat()(2);
  poseIinM.pose.pose.orientation.w = state->_imu->quat()(3);
  poseIinM.pose.pose.position.x = state->_imu->pos()(0);
  poseIinM.pose.pose.position.y = state->_imu->pos()(1);
  poseIinM.pose.pose.position.z = state->_imu->pos()(2);

  // Finally set the covariance in the message (in the order position then orientation as per ros convention)
  std::vector<std::shared_ptr<Type>> statevars;
  statevars.push_back(state->_imu->pose()->p());
  statevars.push_back(state->_imu->pose()->q());
  Eigen::Matrix<double, 6, 6> covariance_posori = StateHelper::get_marginal_covariance(_app->get_state(), statevars);
  covariance_posori = ros_pose_covariance(covariance_posori, state->_imu->Rot());
  for (int r = 0; r < 6; r++) {
    for (int c = 0; c < 6; c++) {
      poseIinM.pose.covariance[6 * r + c] = covariance_posori(r, c);
    }
  }
  pub_poseimu->publish(poseIinM);

  //=========================================================
  //=========================================================

  // Append to our pose vector
  geometry_msgs::msg::PoseStamped posetemp;
  posetemp.header = poseIinM.header;
  posetemp.pose = poseIinM.pose.pose;
  if (path_history_size > 0)
    poses_imu.push_back(posetemp);
  while (poses_imu.size() > static_cast<size_t>(path_history_size))
    poses_imu.pop_front();

  if (pub_pathimu->get_subscription_count() != 0) {
    nav_msgs::msg::Path path;
    path.header = poseIinM.header;
    path.poses.assign(poses_imu.begin(), poses_imu.end());
    pub_pathimu->publish(path);
  }
}

void ROS2Visualizer::publish_images() {

  const auto state = _app->get_state();
  if (!state || track_image_publish_rate == 0 || it_pub_tracks.getNumSubscribers() == 0)
    return;
  // Tracking can precede initialization or propagation; its pixels belong to the camera input,
  // not to the (possibly older) estimator pose. Preserve the camera acquisition clock.
  const double camera_time = last_camera_time >= 0 ? last_camera_time : state->_timestamp;
  if (camera_time < 0 ||
      (last_visualization_timestamp_image >= 0 && camera_time - last_visualization_timestamp_image < 1.0 / track_image_publish_rate - 1e-6))
    return;

  // Get our image of history tracks
  cv::Mat img_history = _app->get_historical_viz_image();
  if (img_history.empty())
    return;
  last_visualization_timestamp_image = camera_time;

  // Create our message
  std_msgs::msg::Header header;
  header.stamp = ROSVisualizerHelper::get_time_from_seconds(camera_time);
  header.frame_id = camera_frame_prefix + "0";
  sensor_msgs::msg::Image::SharedPtr exl_msg = cv_bridge::CvImage(header, "bgr8", img_history).toImageMsg();

  // Publish
  it_pub_tracks.publish(exl_msg);
}

void ROS2Visualizer::publish_features() {
  auto publish = [&](const auto &publisher, const auto &get_features) {
    if (publisher->get_subscription_count() == 0)
      return;
    auto cloud = ROSVisualizerHelper::get_ros_pointcloud(_node, get_features());
    cloud.header.stamp = measurement_stamp();
    cloud.header.frame_id = global_frame;
    publisher->publish(cloud);
  };
  publish(pub_points_msckf, [&] { return _app->get_good_features_MSCKF(); });
  publish(pub_points_slam, [&] { return _app->get_features_SLAM(); });
  publish(pub_points_aruco, [&] { return _app->get_features_ARUCO(); });
  if (_sim)
    publish(pub_points_sim, [&] { return _sim->get_map_vec(); });
}

void ROS2Visualizer::publish_groundtruth() {

  // Our groundtruth state
  Eigen::Matrix<double, 17, 1> state_gt;

  // We want to publish in the IMU clock frame
  // The timestamp in the state will be the last camera time
  double t_ItoC = _app->get_state()->_calib_dt_CAMtoIMU->value()(0);
  double timestamp_inI = _app->get_state()->_timestamp + t_ItoC;

  // Check that we have the timestamp in our GT file [time(sec),q_GtoI,p_IinG,v_IinG,b_gyro,b_accel]
  if (_sim == nullptr && (gt_states.empty() || !DatasetReader::get_gt_state(timestamp_inI, state_gt, gt_states))) {
    return;
  }

  // Get the simulated groundtruth
  // NOTE: we get the true time in the IMU clock frame
  if (_sim != nullptr) {
    timestamp_inI = _app->get_state()->_timestamp + _sim->get_true_parameters().calib_camimu_dt;
    if (!_sim->get_state(timestamp_inI, state_gt))
      return;
  }

  // Get the GT and system state state
  Eigen::Matrix<double, 16, 1> state_ekf = _app->get_state()->_imu->value();

  // Create pose of IMU
  geometry_msgs::msg::PoseStamped poseIinM;
  poseIinM.header.stamp = ROSVisualizerHelper::get_time_from_seconds(timestamp_inI);
  poseIinM.header.frame_id = global_frame;
  poseIinM.pose.orientation.x = state_gt(1, 0);
  poseIinM.pose.orientation.y = state_gt(2, 0);
  poseIinM.pose.orientation.z = state_gt(3, 0);
  poseIinM.pose.orientation.w = state_gt(4, 0);
  poseIinM.pose.position.x = state_gt(5, 0);
  poseIinM.pose.position.y = state_gt(6, 0);
  poseIinM.pose.position.z = state_gt(7, 0);
  pub_posegt->publish(poseIinM);

  // Append to our pose vector
  if (path_history_size > 0)
    poses_gt.push_back(poseIinM);
  while (poses_gt.size() > static_cast<size_t>(path_history_size))
    poses_gt.pop_front();

  if (pub_pathgt->get_subscription_count() != 0) {
    nav_msgs::msg::Path path;
    path.header = poseIinM.header;
    path.poses.assign(poses_gt.begin(), poses_gt.end());
    pub_pathgt->publish(path);
  }

  // Publish our transform on TF
  geometry_msgs::msg::TransformStamped trans;
  trans.header.stamp = poseIinM.header.stamp;
  trans.header.frame_id = global_frame;
  trans.child_frame_id = truth_frame;
  trans.transform.rotation.x = state_gt(1, 0);
  trans.transform.rotation.y = state_gt(2, 0);
  trans.transform.rotation.z = state_gt(3, 0);
  trans.transform.rotation.w = state_gt(4, 0);
  trans.transform.translation.x = state_gt(5, 0);
  trans.transform.translation.y = state_gt(6, 0);
  trans.transform.translation.z = state_gt(7, 0);
  if (publish_global2imu_tf) {
    mTfBr->sendTransform(trans);
  }

  //==========================================================================
  //==========================================================================

  // Difference between positions
  double dx = state_ekf(4, 0) - state_gt(5, 0);
  double dy = state_ekf(5, 0) - state_gt(6, 0);
  double dz = state_ekf(6, 0) - state_gt(7, 0);
  double err_pos = std::sqrt(dx * dx + dy * dy + dz * dz);

  // Quaternion error
  Eigen::Matrix<double, 4, 1> quat_gt, quat_st, quat_diff;
  quat_gt << state_gt(1, 0), state_gt(2, 0), state_gt(3, 0), state_gt(4, 0);
  quat_st << state_ekf(0, 0), state_ekf(1, 0), state_ekf(2, 0), state_ekf(3, 0);
  quat_diff = quat_multiply(quat_st, Inv(quat_gt));
  double err_ori = (180 / M_PI) * 2 * quat_diff.block(0, 0, 3, 1).norm();

  //==========================================================================
  //==========================================================================

  // Get covariance of pose
  std::vector<std::shared_ptr<Type>> statevars;
  statevars.push_back(_app->get_state()->_imu->q());
  statevars.push_back(_app->get_state()->_imu->p());
  Eigen::Matrix<double, 6, 6> covariance = StateHelper::get_marginal_covariance(_app->get_state(), statevars);

  // Calculate NEES values
  // NOTE: need to manually multiply things out to make static asserts work
  // NOTE: https://github.com/rpng/open_vins/pull/226
  // NOTE: https://github.com/rpng/open_vins/issues/236
  // NOTE: https://gitlab.com/libeigen/eigen/-/issues/1664
  Eigen::Vector3d quat_diff_vec = quat_diff.block(0, 0, 3, 1);
  Eigen::Vector3d cov_vec = covariance.block(0, 0, 3, 3).inverse() * 2 * quat_diff.block(0, 0, 3, 1);
  double ori_nees = 2 * quat_diff_vec.dot(cov_vec);
  Eigen::Vector3d errpos = state_ekf.block(4, 0, 3, 1) - state_gt.block(5, 0, 3, 1);
  double pos_nees = errpos.transpose() * covariance.block(3, 3, 3, 3).inverse() * errpos;

  //==========================================================================
  //==========================================================================

  // Update our average variables
  if (!std::isnan(ori_nees) && !std::isnan(pos_nees)) {
    summed_mse_ori += err_ori * err_ori;
    summed_mse_pos += err_pos * err_pos;
    summed_nees_ori += ori_nees;
    summed_nees_pos += pos_nees;
    summed_number++;
  }

  // Nice display for the user
  PRINT_INFO(REDPURPLE "error to gt => %.3f, %.3f (deg,m) | rmse => %.3f, %.3f (deg,m) | called %d times\n" RESET, err_ori, err_pos,
             std::sqrt(summed_mse_ori / summed_number), std::sqrt(summed_mse_pos / summed_number), (int)summed_number);
  PRINT_INFO(REDPURPLE "nees => %.1f, %.1f (ori,pos) | avg nees = %.1f, %.1f (ori,pos)\n" RESET, ori_nees, pos_nees,
             summed_nees_ori / summed_number, summed_nees_pos / summed_number);

  //==========================================================================
  //==========================================================================
}

bool ROS2Visualizer::has_loopclosure_subscribers() const {
  return pub_loop_pose->get_subscription_count() != 0 || pub_loop_point->get_subscription_count() != 0 ||
         pub_loop_extrinsic->get_subscription_count() != 0 || pub_loop_intrinsics->get_subscription_count() != 0 ||
         it_pub_loop_img_depth.getNumSubscribers() != 0 || it_pub_loop_img_depth_color.getNumSubscribers() != 0;
}

void ROS2Visualizer::publish_loopclosure_information() {
  if (!has_loopclosure_subscribers())
    return;

  // Get the current tracks in this frame
  double active_tracks_time1 = -1;
  double active_tracks_time2 = -1;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_posinG;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_uvd;
  cv::Mat active_cam0_image;
  _app->get_active_tracks(active_tracks_time1, active_tracks_posinG, active_tracks_uvd);
  _app->get_active_image(active_tracks_time2, active_cam0_image);
  // A subscriber can connect after this frame's drawing decision; wait for the next frame.
  if (active_cam0_image.empty())
    return;
  if (active_tracks_time1 == -1)
    return;
  if (_app->get_state()->_clones_IMU.find(active_tracks_time1) == _app->get_state()->_clones_IMU.end())
    return;
  if (active_tracks_time1 != active_tracks_time2)
    return;

  // Default header
  std_msgs::msg::Header header;
  header.stamp = ROSVisualizerHelper::get_time_from_seconds(active_tracks_time1 + _app->get_state()->_calib_dt_CAMtoIMU->value()(0));

  //======================================================
  // Check if we have subscribers for the pose odometry, camera intrinsics, or extrinsics
  if (pub_loop_pose->get_subscription_count() != 0 || pub_loop_extrinsic->get_subscription_count() != 0 ||
      pub_loop_intrinsics->get_subscription_count() != 0) {

    // PUBLISH HISTORICAL POSE ESTIMATE
    nav_msgs::msg::Odometry odometry_pose;
    odometry_pose.header = header;
    odometry_pose.header.frame_id = global_frame;
    odometry_pose.child_frame_id = imu_frame;
    odometry_pose.pose.pose.position.x = _app->get_state()->_clones_IMU.at(active_tracks_time1)->pos()(0);
    odometry_pose.pose.pose.position.y = _app->get_state()->_clones_IMU.at(active_tracks_time1)->pos()(1);
    odometry_pose.pose.pose.position.z = _app->get_state()->_clones_IMU.at(active_tracks_time1)->pos()(2);
    odometry_pose.pose.pose.orientation.x = _app->get_state()->_clones_IMU.at(active_tracks_time1)->quat()(0);
    odometry_pose.pose.pose.orientation.y = _app->get_state()->_clones_IMU.at(active_tracks_time1)->quat()(1);
    odometry_pose.pose.pose.orientation.z = _app->get_state()->_clones_IMU.at(active_tracks_time1)->quat()(2);
    odometry_pose.pose.pose.orientation.w = _app->get_state()->_clones_IMU.at(active_tracks_time1)->quat()(3);
    pub_loop_pose->publish(odometry_pose);

    // PUBLISH IMU TO CAMERA0 EXTRINSIC
    // need to flip the transform to the IMU frame
    Eigen::Vector4d q_ItoC = _app->get_state()->_calib_IMUtoCAM.at(0)->quat();
    Eigen::Vector3d p_CinI = -_app->get_state()->_calib_IMUtoCAM.at(0)->Rot().transpose() * _app->get_state()->_calib_IMUtoCAM.at(0)->pos();
    nav_msgs::msg::Odometry odometry_calib;
    odometry_calib.header = header;
    odometry_calib.header.frame_id = imu_frame;
    odometry_calib.child_frame_id = camera_frame_prefix + "0";
    odometry_calib.pose.pose.position.x = p_CinI(0);
    odometry_calib.pose.pose.position.y = p_CinI(1);
    odometry_calib.pose.pose.position.z = p_CinI(2);
    odometry_calib.pose.pose.orientation.x = q_ItoC(0);
    odometry_calib.pose.pose.orientation.y = q_ItoC(1);
    odometry_calib.pose.pose.orientation.z = q_ItoC(2);
    odometry_calib.pose.pose.orientation.w = q_ItoC(3);
    pub_loop_extrinsic->publish(odometry_calib);

    // PUBLISH CAMERA0 INTRINSICS
    bool is_fisheye = (std::dynamic_pointer_cast<ov_core::CamEqui>(_app->get_params().camera_intrinsics.at(0)) != nullptr);
    sensor_msgs::msg::CameraInfo cameraparams;
    cameraparams.header = header;
    cameraparams.header.frame_id = camera_frame_prefix + "0";
    cameraparams.width = active_cam0_image.cols;
    cameraparams.height = active_cam0_image.rows;
    cameraparams.distortion_model = is_fisheye ? "equidistant" : "plumb_bob";
    Eigen::VectorXd cparams = _app->get_state()->_cam_intrinsics.at(0)->value();
    cameraparams.d = {cparams(4), cparams(5), cparams(6), cparams(7)};
    if (!is_fisheye)
      cameraparams.d.push_back(0.0); // plumb_bob includes the fifth radial coefficient k3.
    cameraparams.k = {cparams(0), 0, cparams(2), 0, cparams(1), cparams(3), 0, 0, 1};
    cameraparams.r = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    cameraparams.p = {cparams(0), 0, cparams(2), 0, 0, cparams(1), cparams(3), 0, 0, 0, 1, 0};
    pub_loop_intrinsics->publish(cameraparams);
  }

  //======================================================
  // PUBLISH FEATURE TRACKS IN THE GLOBAL FRAME OF REFERENCE
  if (pub_loop_point->get_subscription_count() != 0) {

    // Construct the message
    sensor_msgs::msg::PointCloud point_cloud;
    point_cloud.header = header;
    point_cloud.header.frame_id = global_frame;
    for (const auto &feattimes : active_tracks_posinG) {

      // Get this feature information
      size_t featid = feattimes.first;
      Eigen::Vector3d uvd = Eigen::Vector3d::Zero();
      if (active_tracks_uvd.find(featid) != active_tracks_uvd.end()) {
        uvd = active_tracks_uvd.at(featid);
      }
      Eigen::Vector3d pFinG = active_tracks_posinG.at(featid);

      // Push back 3d point
      geometry_msgs::msg::Point32 p;
      p.x = pFinG(0);
      p.y = pFinG(1);
      p.z = pFinG(2);
      point_cloud.points.push_back(p);

      // Push back the uv_norm, uv_raw, and feature id
      // NOTE: we don't use the normalized coordinates to save time here
      // NOTE: they will have to be re-normalized in the loop closure code
      sensor_msgs::msg::ChannelFloat32 p_2d;
      p_2d.values.push_back(0);
      p_2d.values.push_back(0);
      p_2d.values.push_back(uvd(0));
      p_2d.values.push_back(uvd(1));
      p_2d.values.push_back(featid);
      point_cloud.channels.push_back(p_2d);
    }
    pub_loop_point->publish(point_cloud);
  }

  //======================================================
  // Depth images of sparse points and its colorized version
  if (it_pub_loop_img_depth.getNumSubscribers() != 0 || it_pub_loop_img_depth_color.getNumSubscribers() != 0) {

    // Create the images we will populate with the depths
    std::pair<int, int> wh_pair = {active_cam0_image.cols, active_cam0_image.rows};
    cv::Mat depthmap = cv::Mat::zeros(wh_pair.second, wh_pair.first, CV_16UC1);
    cv::Mat depthmap_viz;
    if (active_cam0_image.channels() == 1)
      cv::cvtColor(active_cam0_image, depthmap_viz, cv::COLOR_GRAY2BGR);
    else
      depthmap_viz = active_cam0_image.clone();

    // Loop through all points and append
    for (const auto &feattimes : active_tracks_uvd) {

      // Get this feature information
      size_t featid = feattimes.first;
      Eigen::Vector3d uvd = active_tracks_uvd.at(featid);

      // Skip invalid points
      double dw = 4;
      if (!uvd.allFinite() || uvd(2) <= 0 || uvd(2) > std::numeric_limits<uint16_t>::max() / 1000.0 || uvd(0) < dw ||
          uvd(0) > wh_pair.first - dw || uvd(1) < dw || uvd(1) > wh_pair.second - dw) {
        continue;
      }

      // Append the depth
      // NOTE: scaled by 1000 to fit the 16U
      // NOTE: access order is y,x (stupid opencv convention stuff)
      depthmap.at<uint16_t>((int)uvd(1), (int)uvd(0)) = (uint16_t)(1000 * uvd(2));

      // Taken from LSD-SLAM codebase segment into 0-4 meter segments:
      // https://github.com/tum-vision/lsd_slam/blob/d1e6f0e1a027889985d2e6b4c0fe7a90b0c75067/lsd_slam_core/src/util/globalFuncs.cpp#L87-L96
      float id = 1.0f / (float)uvd(2);
      float r = (0.0f - id) * 255 / 1.0f;
      if (r < 0)
        r = -r;
      float g = (1.0f - id) * 255 / 1.0f;
      if (g < 0)
        g = -g;
      float b = (2.0f - id) * 255 / 1.0f;
      if (b < 0)
        b = -b;
      uchar rc = r < 0 ? 0 : (r > 255 ? 255 : r);
      uchar gc = g < 0 ? 0 : (g > 255 ? 255 : g);
      uchar bc = b < 0 ? 0 : (b > 255 ? 255 : b);
      cv::Scalar color(255 - rc, 255 - gc, 255 - bc);

      // Small square around the point (note the above bound check needs to take into account this width)
      cv::Point p0(uvd(0) - dw, uvd(1) - dw);
      cv::Point p1(uvd(0) + dw, uvd(1) + dw);
      cv::rectangle(depthmap_viz, p0, p1, color, -1);
    }

    // Create our messages
    header.frame_id = camera_frame_prefix + "0";
    sensor_msgs::msg::Image::SharedPtr exl_msg1 =
        cv_bridge::CvImage(header, sensor_msgs::image_encodings::TYPE_16UC1, depthmap).toImageMsg();
    it_pub_loop_img_depth.publish(exl_msg1);
    sensor_msgs::msg::Image::SharedPtr exl_msg2 = cv_bridge::CvImage(header, "bgr8", depthmap_viz).toImageMsg();
    it_pub_loop_img_depth_color.publish(exl_msg2);
  }
}
