/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef OV_MSCKF_ROS2_SENSOR_QOS_H
#define OV_MSCKF_ROS2_SENSOR_QOS_H

#include <rclcpp/qos_overriding_options.hpp>

namespace ov_msckf {

/// Allow the standard qos_overrides.<resolved topic>.subscription parameters.
/// Keep sensor histories bounded even when an external parameter file overrides depth.
inline rclcpp::QosOverridingOptions sensor_qos_overrides(size_t maximum_depth) {
  return {{rclcpp::QosPolicyKind::Depth, rclcpp::QosPolicyKind::Reliability}, [maximum_depth](const rclcpp::QoS &qos) {
            rclcpp::QosCallbackResult result;
            result.successful = qos.history() == rclcpp::HistoryPolicy::KeepLast && qos.depth() > 0 && qos.depth() <= maximum_depth;
            if (!result.successful)
              result.reason = "Sensor QoS requires keep_last with depth in [1, " + std::to_string(maximum_depth) + "]";
            return result;
          }};
}

} // namespace ov_msckf
#endif
