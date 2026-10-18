#!/usr/bin/env python3
"""Verify idle fast prediction and late consumers on remapped TF/odometry topics."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from tf2_msgs.msg import TFMessage


def stop(process):
    if process is not None and process.poll() is None:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', type=Path, required=True)
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--domain-id', type=int, default=154)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prefix = args.install.resolve() / 'ov_msckf'
    share = prefix / 'share/ov_msckf'
    os.environ.update(ROS_DOMAIN_ID=str(args.domain_id), ROS_LOG_DIR=str(args.output.resolve() / 'ros-logs'))
    rclpy.init(args=[])
    node = rclpy.create_node('open_vins_demand_audit')
    records = []

    def run_case(late):
        name = 'late-consumers' if late else 'headless'
        out = args.output / name
        out.mkdir()
        counts = {'pose': 0, 'tf': 0, 'odometry': 0}
        errors = []
        def check(condition, message):
            if not condition and message not in errors:
                errors.append(message)
        def sensor_header(message):
            check(1e9 < message.header.stamp.sec < 1.6e9, 'Output did not use EuRoC sensor time')
        def pose(message):
            counts['pose'] += 1
            sensor_header(message)
            p = message.pose.pose.position
            check(all(math.isfinite(v) for v in (p.x, p.y, p.z)), 'Nonfinite pose')
        def transforms(message):
            for transform in message.transforms:
                if transform.child_frame_id != 'imu':
                    continue
                counts['tf'] += 1
                sensor_header(transform)
                check(transform.header.frame_id == 'global', 'Wrong TF parent frame')
                q = transform.transform.rotation
                check(abs(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w - 1) < 1e-6, 'Invalid TF quaternion')
        def odometry(message):
            counts['odometry'] += 1
            sensor_header(message)
            check(message.header.frame_id == 'global' and message.child_frame_id == 'imu', 'Wrong odometry frames')
            check(all(math.isfinite(v) for v in (*message.pose.covariance, *message.twist.covariance)), 'Nonfinite covariance')
        pose_sub = node.create_subscription(PoseWithCovarianceStamped, '/poseimu', pose, 10)
        tf_sub = odom_sub = None
        duration = 20 if late else 10
        command = [str(prefix / 'lib/ov_msckf/run_subscribe_msckf'), str(share / 'config/euroc_mav/estimator_config.yaml'),
                   '--ros-args', '--params-file', str(share / 'config/euroc_mav/ros2_euroc_throughput_full.yaml'),
                   '-r', '/tf:=/ov_demand/tf', '-r', 'odomimu:=/ov_demand/odom']
        playback = ['ros2', 'bag', 'play', str(args.bag.resolve()), '--clock', '100', '--rate', '1',
                    '--playback-duration', str(duration), '--delay', '1', '--disable-keyboard-controls',
                    '--read-ahead-queue-size', '200', '--progress-bar-update-rate', '0',
                    '--qos-profile-overrides-path', str(share / 'config/euroc_mav/ros2_player.yaml'),
                    '--wait-for-all-acked', '5000', '--topics', '/imu0', '/cam0/image_raw', '/cam1/image_raw']
        (out / 'commands.json').write_text(json.dumps([command, playback], indent=2) + '\n')
        process = player = None
        try:
            with (out / 'node.log').open('w') as log, (out / 'player.log').open('w') as playlog:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + 12
                while not all(node.get_subscriptions_info_by_topic(topic) for topic in ('/imu0', '/cam0/image_raw', '/cam1/image_raw')):
                    check(process.poll() is None, 'Node exited during startup')
                    assert process.poll() is None and time.monotonic() < deadline
                    rclpy.spin_once(node, timeout_sec=.05)
                player = subprocess.Popen(playback, stdout=playlog, stderr=subprocess.STDOUT, start_new_session=True)
                start = time.monotonic()
                tf_started = tf_finished = odom_started = odom_finished = False
                while player.poll() is None:
                    assert process.poll() is None
                    elapsed = time.monotonic() - start
                    assert elapsed < duration + 20, 'Playback did not finish'
                    if late and elapsed >= 8 and not tf_started:
                        tf_sub = node.create_subscription(TFMessage, '/ov_demand/tf', transforms, qos_profile_sensor_data)
                        tf_started = True
                    if late and elapsed >= 12 and not tf_finished:
                        node.destroy_subscription(tf_sub)
                        tf_sub = None
                        tf_finished = True
                    if late and elapsed >= 14 and not odom_started:
                        odom_sub = node.create_subscription(Odometry, '/ov_demand/odom', odometry, 100)
                        odom_started = True
                    if late and elapsed >= 18 and not odom_finished:
                        node.destroy_subscription(odom_sub)
                        odom_sub = None
                        odom_finished = True
                    rclpy.spin_once(node, timeout_sec=.01)
                assert player.returncode == 0
                stop(process)
                assert process.returncode == 0, 'Node did not shut down cleanly'
        finally:
            stop(player)
            stop(process)
            for subscription in (pose_sub, tf_sub, odom_sub):
                if subscription is not None:
                    node.destroy_subscription(subscription)
        text = (out / 'node.log').read_text()
        values = re.findall(r'ROS2 outputs: fast_odometry=(\d+)', text)
        assert len(values) == 1, 'Missing prediction count'
        fast = int(values[0])
        check(counts['pose'] > 10, 'No initialized visual-inertial poses')
        if late:
            check(counts['tf'] > 10 and counts['odometry'] > 10, 'Late remapped consumers did not receive output')
            check(10 < fast < 2000, 'Fast prediction did not follow the consumer windows')
        else:
            check(fast == 0, 'Headless node computed unused fast predictions')
        record = {'case': name, 'passed': not errors, 'counts': counts, 'fast_odometry_updates': fast,
                  'errors': errors, 'shutdown_returncode': process.returncode}
        records.append(record)
        (out / 'results.json').write_text(json.dumps(record, indent=2) + '\n')
        assert not errors, record
        print('Passed demand audit:', name, counts, 'fast predictions', fast, flush=True)

    try:
        run_case(False)
        run_case(True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        (args.output / 'results.json').write_text(json.dumps({'passed': len(records) == 2 and all(r['passed'] for r in records),
                                                            'records': records}, indent=2) + '\n')


if __name__ == '__main__':
    main()
