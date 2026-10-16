/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef OV_CORE_ROS2_PARAMETERS_H
#define OV_CORE_ROS2_PARAMETERS_H

#include <limits>
#include <rclcpp/rclcpp.hpp>
#include <stdexcept>
#include <type_traits>

namespace ov_core {

/// Startup settings are visible through parameter services and cannot silently change after loading.
template <typename T> T startup_parameter(rclcpp::Node &node, const std::string &name, const T &default_value) {
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.read_only = true;
  if (!node.has_parameter(name))
    node.declare_parameter(name, rclcpp::ParameterValue(default_value), descriptor);
  const auto value = node.get_parameter(name).get_value<T>();
  if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool> && sizeof(T) < sizeof(int64_t)) {
    if (value < static_cast<int64_t>(std::numeric_limits<T>::lowest()) || value > static_cast<int64_t>(std::numeric_limits<T>::max()))
      throw std::invalid_argument(name + " does not fit its declared integer type");
  }
  if (!node.describe_parameter(name).read_only)
    throw std::invalid_argument(name + ": disable automatically_declare_parameters_from_overrides for startup parameters");
  return value;
}

} // namespace ov_core
#endif
