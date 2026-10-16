#!/usr/bin/env python3
"""Check live ROS 2 outputs while replaying a EuRoC recording.

Source the underlay and overlay, then start OpenVINS with ROS parameters:
  global_frame_id:=audit_world imu_frame_id:=audit_imu
  camera_frame_prefix:=audit_cam path_history_size:=5 track_image_publish_rate:=10.0
Run this observer in the same isolated ROS_DOMAIN_ID before starting ros2 bag play.
The timestamp range deliberately distinguishes EuRoC sensor time from wall time.
"""
import argparse
import json
import math
from pathlib import Path
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path as RosPath
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_msgs.msg import TFMessage

parser = argparse.ArgumentParser()
parser.add_argument('--seconds', type=float, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
rclpy.init()
node = rclpy.create_node('open_vins_output_contract_audit')
counts = dict(pose=0, odometry=0, path=0, image=0, points=0, tf=0,
              loop_pose=0, intrinsics=0, depth=0, depth_color=0)
errors = []
last = {}
first = {}
peak_path = 0

def check(condition, error):
    if not condition and error not in errors:
        errors.append(error)

def header(message, kind, frame='audit_world'):
    stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
    check(1e9 < stamp < 1.6e9, kind + ': header did not use EuRoC sensor time')
    check(message.header.frame_id == frame, kind + ': unexpected frame ' + message.header.frame_id)
    if kind in last:
        check(stamp >= last[kind], kind + ': timestamp went backwards')
    first.setdefault(kind, stamp)
    last[kind] = stamp
    counts[kind] += 1

def pose(message):
    header(message, 'pose')
    check(all(math.isfinite(x) for x in message.pose.covariance), 'pose: nonfinite covariance')

def odometry(message):
    header(message, 'odometry')
    check(message.child_frame_id == 'audit_imu', 'odometry: wrong twist frame')
    check(all(math.isfinite(x) for x in message.twist.covariance), 'odometry: nonfinite twist covariance')

def path(message):
    global peak_path
    header(message, 'path')
    peak_path = max(peak_path, len(message.poses))
    check(len(message.poses) <= 5, 'path: exceeded configured history bound')
    check(all(p.header.frame_id == 'audit_world' for p in message.poses), 'path: pose frame mismatch')

def loop_pose(message):
    header(message, 'loop_pose')
    check(message.child_frame_id == 'audit_imu', 'loop pose: wrong child frame')

def intrinsics(message):
    header(message, 'intrinsics', 'audit_cam0')
    check(message.width == 752 and message.height == 480, 'CameraInfo: wrong image dimensions')
    check(message.k[0] > 0 and message.k[4] > 0 and message.k[8] == 1, 'CameraInfo: invalid K')
    check(list(message.r) == [1, 0, 0, 0, 1, 0, 0, 0, 1], 'CameraInfo: invalid R')
    check(message.p[0] > 0 and message.p[5] > 0 and message.p[10] == 1, 'CameraInfo: invalid P')
    check(message.distortion_model == 'plumb_bob' and len(message.d) == 5, 'CameraInfo: invalid distortion schema')

def depth(message, colored):
    kind = 'depth_color' if colored else 'depth'
    header(message, kind, 'audit_cam0')
    check(message.encoding == ('bgr8' if colored else '16UC1'), kind + ': wrong encoding')
    check(message.step >= message.width * (3 if colored else 2), kind + ': wrong stride')
    check(len(message.data) == message.step * message.height, kind + ': inconsistent pixel storage')

def transforms(message):
    for transform in message.transforms:
        if transform.child_frame_id == 'audit_imu':
            header(transform, 'tf')
        elif transform.child_frame_id.startswith('audit_cam'):
            check(transform.header.frame_id == 'audit_imu', 'camera tf: parent frame mismatch')

subscriptions = [
    node.create_subscription(Odometry, '/loop_pose', loop_pose, 10),
    node.create_subscription(CameraInfo, '/loop_intrinsics', intrinsics, 10),
    node.create_subscription(Image, '/loop_depth', lambda m: depth(m, False), qos_profile_sensor_data),
    node.create_subscription(Image, '/loop_depth_colored', lambda m: depth(m, True), qos_profile_sensor_data),
    node.create_subscription(PoseWithCovarianceStamped, '/poseimu', pose, 10),
    node.create_subscription(Odometry, '/odomimu', odometry, 100),
    node.create_subscription(RosPath, '/pathimu', path, 10),
    node.create_subscription(Image, '/trackhist', lambda m: header(m, 'image', 'audit_cam0'), qos_profile_sensor_data),
    node.create_subscription(PointCloud2, '/points_msckf', lambda m: header(m, 'points'), qos_profile_sensor_data),
    node.create_subscription(TFMessage, '/tf', transforms, qos_profile_sensor_data),
]
deadline = time.monotonic() + args.seconds
while time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.05)
node.destroy_node()
rclpy.shutdown()
for kind, count in counts.items():
    check(count > 0, 'No output received for ' + kind)
if counts['image'] > 1:
    check(counts['image'] <= (last['image'] - first['image']) * 10 + 2, 'image: exceeded configured publication rate')
report = dict(counts=counts, peak_path=peak_path, errors=errors)
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
assert not errors, errors
