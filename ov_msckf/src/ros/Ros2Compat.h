/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef OV_MSCKF_ROS2_COMPAT_H
#define OV_MSCKF_ROS2_COMPAT_H

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>

namespace ov_msckf {

// Jazzy only accepts rmw_qos_profile_t here. Lyrical prefers rclcpp::QoS (and drops the rmw overloads for
// message filters), so detect the available signature instead of keying off distribution versions.

template <typename Node, typename = void> struct has_qos_image_publisher : std::false_type {};
template <typename Node>
struct has_qos_image_publisher<Node,
                               std::void_t<decltype(image_transport::create_publisher(
                                   std::declval<Node &>(), std::declval<const std::string &>(), std::declval<const rclcpp::QoS &>()))>>
    : std::true_type {};

template <typename Node> image_transport::Publisher create_image_publisher(Node &node, const std::string &topic, const rclcpp::QoS &qos) {
  if constexpr (has_qos_image_publisher<Node>::value)
    return image_transport::create_publisher(node, topic, qos);
  else
    return image_transport::create_publisher(&node, topic, qos.get_rmw_qos_profile());
}

template <typename Subscriber, typename NodePtr>
std::shared_ptr<Subscriber> create_filter_subscriber(const NodePtr &node, const std::string &topic, const rclcpp::QoS &qos,
                                                     const rclcpp::SubscriptionOptions &options) {
  if constexpr (std::is_constructible_v<Subscriber, const NodePtr &, const std::string &, const rclcpp::QoS &,
                                        const rclcpp::SubscriptionOptions &>)
    return std::make_shared<Subscriber>(node, topic, qos, options);
  else
    return std::make_shared<Subscriber>(node, topic, qos.get_rmw_qos_profile(), options);
}

} // namespace ov_msckf

#endif // OV_MSCKF_ROS2_COMPAT_H
