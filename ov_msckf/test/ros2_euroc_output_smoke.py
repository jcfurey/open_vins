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
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_msgs.msg import TFMessage

parser = argparse.ArgumentParser()
parser.add_argument('--seconds', type=float, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--camera-width', type=int, default=752)
parser.add_argument('--camera-height', type=int, default=480)
parser.add_argument('--image-reliability', choices=['reliable','best_effort'], default='reliable')
parser.add_argument('--calibration-tf', choices=['static', 'dynamic'], default='static')
args = parser.parse_args()
if args.camera_width <= 0 or args.camera_height <= 0:
    parser.error('camera dimensions must be positive')
rclpy.init()
node = rclpy.create_node('open_vins_output_contract_audit')
image_qos = QoSProfile(depth=10, reliability=(ReliabilityPolicy.RELIABLE if args.image_reliability == 'reliable'
                                           else ReliabilityPolicy.BEST_EFFORT))
counts = dict(pose=0, odometry=0, path=0, image=0, points=0, tf=0,
              loop_pose=0, intrinsics=0, depth=0, depth_color=0)
if args.calibration_tf == 'static':
    counts.update(tf_static=0, tf_static_late=0)
camera_frames = set()
late_camera_frames = set()
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
    check(message.width == args.camera_width and message.height == args.camera_height,
          'CameraInfo: wrong image dimensions')
    check(message.k[0] > 0 and message.k[4] > 0 and message.k[8] == 1, 'CameraInfo: invalid K')
    # Online calibration may refine EuRoC's supplied focal lengths, but the
    # published calibration must use the processed image's pixel coordinates.
    expected_fx = 458.654 * args.camera_width / 752
    expected_fy = 457.296 * args.camera_height / 480
    check(abs(message.k[0] / expected_fx - 1) < .25 and abs(message.k[4] / expected_fy - 1) < .25,
          'CameraInfo: focal lengths do not match processed image scale')
    check(list(message.r) == [1, 0, 0, 0, 1, 0, 0, 0, 1], 'CameraInfo: invalid R')
    check(message.p[0] > 0 and message.p[5] > 0 and message.p[10] == 1, 'CameraInfo: invalid P')
    check(message.distortion_model == 'plumb_bob' and len(message.d) == 5, 'CameraInfo: invalid distortion schema')

def depth(message, colored):
    kind = 'depth_color' if colored else 'depth'
    header(message, kind, 'audit_cam0')
    check(message.encoding == ('bgr8' if colored else '16UC1'), kind + ': wrong encoding')
    check(message.width == args.camera_width and message.height == args.camera_height,
          kind + ': wrong image dimensions')
    check(message.step >= message.width * (3 if colored else 2), kind + ': wrong stride')
    check(len(message.data) == message.step * message.height, kind + ': inconsistent pixel storage')

def transforms(message):
    for transform in message.transforms:
        if transform.child_frame_id == 'audit_imu':
            header(transform, 'tf')
        elif transform.child_frame_id.startswith('audit_cam'):
            check(args.calibration_tf == 'dynamic', 'fixed camera calibration was sent on /tf')
            check(transform.header.frame_id == 'audit_imu', 'camera tf: parent frame mismatch')
            camera_frames.add(transform.child_frame_id)

def static_transforms(message, late=False):
    check(args.calibration_tf == 'static', 'estimated camera calibration was sent on /tf_static')
    for transform in message.transforms:
        check(transform.child_frame_id in ('audit_cam0', 'audit_cam1'), 'static tf: unexpected child frame')
        check(transform.header.frame_id == 'audit_imu', 'static tf: unexpected parent frame')
        q = transform.transform.rotation
        t = transform.transform.translation
        check(all(math.isfinite(v) for v in (q.x, q.y, q.z, q.w, t.x, t.y, t.z)), 'static tf: nonfinite transform')
        check(abs(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w - 1) < 1e-6, 'static tf: invalid rotation')
        (late_camera_frames if late else camera_frames).add(transform.child_frame_id)
        counts['tf_static_late' if late else 'tf_static'] = counts.get('tf_static_late' if late else 'tf_static', 0) + 1

subscriptions = [
    node.create_subscription(Odometry, '/loop_pose', loop_pose, 10),
    node.create_subscription(CameraInfo, '/loop_intrinsics', intrinsics, 10),
    node.create_subscription(Image, '/loop_depth', lambda m: depth(m, False), image_qos),
    node.create_subscription(Image, '/loop_depth_colored', lambda m: depth(m, True), image_qos),
    node.create_subscription(PoseWithCovarianceStamped, '/poseimu', pose, 10),
    node.create_subscription(Odometry, '/odomimu', odometry, 100),
    node.create_subscription(RosPath, '/pathimu', path, 10),
    node.create_subscription(Image, '/trackhist', lambda m: header(m, 'image', 'audit_cam0'), image_qos),
    node.create_subscription(PointCloud2, '/points_msckf', lambda m: header(m, 'points'), qos_profile_sensor_data),
    node.create_subscription(TFMessage, '/tf', transforms, qos_profile_sensor_data),
    node.create_subscription(TFMessage, '/tf_static', static_transforms,
                             QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)),
]
deadline = time.monotonic() + args.seconds
late_subscription = None
first_static_time = None
while time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.05)
    if args.calibration_tf == 'static' and counts['tf_static'] and first_static_time is None:
        first_static_time = time.monotonic()
    if first_static_time is not None and late_subscription is None and time.monotonic() - first_static_time > 1:
        late_subscription = node.create_subscription(
            TFMessage, '/tf_static', lambda m: static_transforms(m, True),
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
node.destroy_node()
rclpy.shutdown()
check(camera_frames == {'audit_cam0', 'audit_cam1'}, 'camera TF tree is incomplete')
if args.calibration_tf == 'static':
    check(late_camera_frames == camera_frames, 'late subscriber did not receive retained calibration')
    check(counts['tf_static'] == 2, 'fixed calibration was republished during the audit')
for kind, count in counts.items():
    check(count > 0, 'No output received for ' + kind)
if counts['image'] > 1:
    check(counts['image'] <= (last['image'] - first['image']) * 10 + 2, 'image: exceeded configured publication rate')
report = dict(counts=counts, peak_path=peak_path, errors=errors,
              expected_camera_dimensions=[args.camera_width, args.camera_height], image_reliability=args.image_reliability)
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
assert not errors, errors
