#!/usr/bin/env python3
"""Exercise installed ROS 2 ingress, parameter overrides, resets, and shutdown.

Run after sourcing the ROS underlay and workspace overlay. Use a private
ROS_DOMAIN_ID when running alongside other ROS applications.
"""

import argparse
from array import array
import math
from pathlib import Path
import re
import signal
import subprocess
import time

import rclpy
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path,
                        help="EuRoC stereo estimator configuration")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-only", action="store_true",
                        help="Use raw image transport only, excluding optional transport plugins from sanitizer checks")
    parser.add_argument("--reliable", action="store_true",
                        help="Verify standard QoS overrides on remapped sensor topics")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    command = [str(args.executable.resolve()), str(args.config.resolve()), "--ros-args",
               "-p", "verbosity:=SILENT", "-p", "topic_imu:=/ov_audit/imu",
               "-p", "topic_camera0:=/ov_audit/left", "-p", "topic_camera1:=/ov_audit/right",
               "-p", "max_camera_queue_size:=3", "-p", "path_history_size:=5"]
    if args.raw_only:
        for topic in ("trackhist", "loop_depth", "loop_depth_colored"):
            command += ["-p", topic + ".enable_pub_plugins:=[image_transport/raw]"]
    if args.reliable:
        for topic in ("/ov_audit/imu", "/ov_audit/left", "/ov_audit/right"):
            command += ["-p", f"qos_overrides.{topic}.subscription.reliability:=reliable",
                        "-p", f"qos_overrides.{topic}.subscription.depth:=20"]
    rclpy.init()
    node = rclpy.create_node("open_vins_runtime_smoke")
    qos = QoSProfile(depth=20) if args.reliable else qos_profile_sensor_data
    imu_pub = node.create_publisher(Imu, "/ov_audit/imu", qos)
    left_pub = node.create_publisher(Image, "/ov_audit/left", qos)
    right_pub = node.create_publisher(Image, "/ov_audit/right", qos)
    publishers = (imu_pub, left_pub, right_pub)
    log_path = args.output / "runtime.log"

    def pump(seconds=0.04):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.01)

    def stamp(message, timestamp):
        message.header.stamp.sec = math.floor(timestamp)
        message.header.stamp.nanosec = round((timestamp - math.floor(timestamp)) * 1e9)

    def call(future):
        rclpy.spin_until_future_complete(node, future, timeout_sec=5)
        assert future.done(), "Parameter service timed out"
        return future.result()

    image = Image(height=480, width=752, encoding="mono8", step=752)
    image.data = array("B", [0]) * (image.height * image.width)
    imu = Imu()
    imu.linear_acceleration.z = 9.81
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 12
            while not all(publisher.get_subscription_count() for publisher in publishers):
                assert process.poll() is None, log_path.read_text()[-5000:]
                assert time.monotonic() < deadline, "SensorDataQoS subscriptions or topic aliases failed"
                pump()

            client = AsyncParameterClient(node, "run_subscribe_msckf")
            assert client.wait_for_services(timeout_sec=5)
            if args.reliable:
                for topic in ("/ov_audit/imu", "/ov_audit/left", "/ov_audit/right"):
                    endpoints = [info for info in node.get_subscriptions_info_by_topic(topic)
                                 if info.node_name == "run_subscribe_msckf"]
                    assert len(endpoints) == 1, endpoints
                    assert endpoints[0].qos_profile.reliability == ReliabilityPolicy.RELIABLE
                    assert endpoints[0].qos_profile.depth == 20
                name = "qos_overrides./ov_audit/imu.subscription.reliability"
                assert call(client.describe_parameters([name])).descriptors[0].read_only
            description = call(client.describe_parameters(["max_camera_queue_size", "max_clones", "cam0_T_imu_cam"]))
            assert all(descriptor.read_only for descriptor in description.descriptors), description
            result = call(client.set_parameters([Parameter("max_camera_queue_size", value=100)]))
            assert not result.results[0].successful, "Startup setting accepted an ineffective runtime change"

            # Cameras without IMU must stay bounded, rather than blocking the callback executor.
            for index in range(24):
                stamp(image, 10 + 0.05 * index)
                left_pub.publish(image)
                right_pub.publish(image)
                pump(0.015)
            pump(0.1)

            # A best-effort IMU publisher must connect. Duplicate and invalid samples are rejected.
            for index in range(15):
                stamp(imu, 12 + 0.005 * index)
                imu_pub.publish(imu)
                pump(0.01)
            imu_pub.publish(imu)
            pump()
            imu.angular_velocity.x = float("nan")
            imu_pub.publish(imu)
            pump()
            imu.angular_velocity.x = 0.0
            imu.header.stamp.nanosec = 1_000_000_000
            imu_pub.publish(imu)
            pump()

            # A bag rewind starts a fresh estimator epoch and still accepts subsequent input.
            for index in range(15):
                stamp(imu, 5 + 0.005 * index)
                imu_pub.publish(imu)
                pump(0.01)
            for index in range(3):
                stamp(image, 5 + 0.05 * index)
                left_pub.publish(image)
                right_pub.publish(image)
                pump()
            assert process.poll() is None, log_path.read_text()[-5000:]
            # Leave work pending when SIGINT arrives, rather than testing only an idle shutdown.
            for index in range(20):
                stamp(imu, 5.08 + 0.005 * index)
                imu_pub.publish(imu)
                if index % 10 == 0:
                    stamp(image, 5.15 + 0.05 * (index // 10))
                    left_pub.publish(image)
                    right_pub.publish(image)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            node.destroy_node()
            rclpy.shutdown()
    text = log_path.read_text()
    assert process.returncode == 0, text[-5000:]
    match = re.search(r"ROS2 ingress: (.*)", text)
    assert match, text[-5000:]
    counters = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", match[1])}
    assert counters["imu"] >= 20 and counters["cameras"] >= 12, counters
    assert counters["peak_cameras"] <= 3 and counters["dropped_cameras"] > 0, counters
    assert counters["invalid"] >= 2 and counters["out_of_order"] >= 1, counters
    assert counters["resets"] >= 1, counters

    for value in ("0", "2147483648"):
        with (args.output / f"invalid_queue_{value}.log").open("w") as log:
            invalid = subprocess.run(command + ["-p", f"max_camera_queue_size:={value}"],
                                     stdout=log, stderr=subprocess.STDOUT, timeout=15)
        assert invalid.returncode != 0, "Invalid queue size was accepted"
    for topic, depth in (("/ov_audit/imu", 0), ("/ov_audit/imu", 100001), ("/ov_audit/left", 1001)):
        with (args.output / f"invalid_qos_{topic.rsplit('/', 1)[-1]}_{depth}.log").open("w") as log:
            invalid = subprocess.run(command + ["-p", f"qos_overrides.{topic}.subscription.depth:={depth}"],
                                     stdout=log, stderr=subprocess.STDOUT, timeout=15)
        assert invalid.returncode != 0, "Unbounded or zero QoS depth was accepted"
    print("ROS 2 runtime smoke passed:", counters)


if __name__ == "__main__":
    main()
