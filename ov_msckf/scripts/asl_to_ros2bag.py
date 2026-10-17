#!/usr/bin/env python3
"""Convert an ASL/EuRoC-format sequence (mav0/imu0, mav0/camN) into a ROS 2 bag.

Header stamps and bag receive times are the dataset nanosecond stamps, and messages
are written in timestamp order (IMU first on ties), so `ros2_serial_msckf` replays the
bag through exactly the same estimator call sequence as `asl_serial_msckf` replays the
folder. Run with the ROS 2 environment sourced:

  python3 asl_to_ros2bag.py ~/datasets/euroc/MH_01_easy ~/bags/MH_01_easy --cameras 2
"""

import argparse
import csv
import heapq
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image, Imu


def rows(path):
    with open(path, newline="") as handle:
        for row in csv.reader(handle):
            if row and not row[0].lstrip().startswith("#"):
                yield [field.strip() for field in row]


def stamp(message, nanoseconds):
    message.header.stamp.sec = nanoseconds // 1_000_000_000
    message.header.stamp.nanosec = nanoseconds % 1_000_000_000


def imu_messages(folder, topic, frame_id):
    for row in rows(folder / "imu0" / "data.csv"):
        ns = int(row[0])
        message = Imu()
        stamp(message, ns)
        message.header.frame_id = frame_id
        message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z = map(float, row[1:4])
        message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z = map(float, row[4:7])
        yield ns, 0, topic, message


def image_messages(folder, camera, topic, frame_id):
    directory = folder / f"cam{camera}"
    for row in rows(directory / "data.csv"):
        ns = int(row[0])
        # Keep the stored depth (TUM-VI *_16 sequences are 16-bit) so subscribers convert it as they would live data
        pixels = cv2.imread(str(directory / "data" / row[1]), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        if pixels is None or pixels.dtype not in (np.uint8, np.uint16):
            raise RuntimeError(f"Unable to read an 8 or 16-bit image from {directory / 'data' / row[1]}")
        message = Image()
        stamp(message, ns)
        message.header.frame_id = frame_id
        message.height, message.width = pixels.shape
        message.encoding = "mono8" if pixels.dtype == np.uint8 else "mono16"
        message.is_bigendian = 0
        message.step = message.width * pixels.itemsize
        message.data = pixels.astype(pixels.dtype.newbyteorder("<"), copy=False).tobytes()
        yield ns, 1 + camera, topic, message


def topic_metadata(topic_id, name, type_name):
    try:
        return rosbag2_py.TopicMetadata(id=topic_id, name=name, type=type_name, serialization_format="cdr")
    except TypeError:  # Humble and older take no id
        return rosbag2_py.TopicMetadata(name=name, type=type_name, serialization_format="cdr")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="sequence folder containing mav0/, or mav0/ itself")
    parser.add_argument("output", type=Path, help="bag directory to create")
    parser.add_argument("--cameras", type=int, default=2)
    parser.add_argument("--storage", default="mcap", choices=("mcap", "sqlite3"))
    parser.add_argument("--imu-topic", default="/imu0")
    parser.add_argument("--camera-topic", default="/cam{}/image_raw", help="format string taking the camera index")
    args = parser.parse_args()

    folder = args.dataset / "mav0" if (args.dataset / "mav0").is_dir() else args.dataset
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(args.output), storage_id=args.storage),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    streams = [imu_messages(folder, args.imu_topic, "imu0")]
    writer.create_topic(topic_metadata(0, args.imu_topic, "sensor_msgs/msg/Imu"))
    for camera in range(args.cameras):
        topic = args.camera_topic.format(camera)
        writer.create_topic(topic_metadata(1 + camera, topic, "sensor_msgs/msg/Image"))
        streams.append(image_messages(folder, camera, topic, f"cam{camera}"))

    count = 0
    # Each CSV is time ordered; merge on (stamp, stream) so IMU precedes images at equal stamps.
    for ns, _, topic, message in heapq.merge(*streams, key=lambda item: (item[0], item[1])):
        writer.write(topic, serialize_message(message), ns)
        count += 1
    del writer
    print(f"Wrote {count} messages to {args.output}")


if __name__ == "__main__":
    main()
