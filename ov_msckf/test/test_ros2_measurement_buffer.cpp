/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <thread>

#include "ros/Ros2MeasurementBuffer.h"

using ov_msckf::Ros2MeasurementBuffer;
using Result = Ros2MeasurementBuffer::Result;

namespace {
ov_core::ImuData imu(double time) { return {time, Eigen::Vector3d::Zero(), Eigen::Vector3d(0, 0, 9.81)}; }
ov_core::CameraData camera(double time, int id = 0) {
  return {time, {id}, {cv::Mat::zeros(8, 8, CV_8UC1)}, {cv::Mat::zeros(8, 8, CV_8UC1)}};
}
} // namespace

TEST(Ros2Buffer, RequiresValidLimits) {
  EXPECT_THROW(Ros2MeasurementBuffer(0, 2, 1, 20), std::invalid_argument);
  EXPECT_THROW(Ros2MeasurementBuffer(2, 0, 1, 20), std::invalid_argument);
  EXPECT_THROW(Ros2MeasurementBuffer(2, 2, 0, 20), std::invalid_argument);
  EXPECT_THROW(Ros2MeasurementBuffer(2, 2, 1, 0), std::invalid_argument);
  EXPECT_THROW(Ros2MeasurementBuffer(2, 2, 1, 20, true, 0), std::invalid_argument);
  EXPECT_THROW(Ros2MeasurementBuffer(2, 2, 1, 20, true, std::numeric_limits<double>::infinity()), std::invalid_argument);
}

TEST(Ros2Buffer, RejectsBadImuAndDuplicates) {
  Ros2MeasurementBuffer buffer(8, 2, 1, 20);
  auto bad = imu(1);
  bad.wm(0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(buffer.push_imu(bad), Result::Invalid);
  EXPECT_EQ(buffer.push_imu(imu(-1)), Result::Invalid);
  EXPECT_EQ(buffer.push_imu(imu(1)), Result::Accepted);
  EXPECT_EQ(buffer.push_imu(imu(1)), Result::OutOfOrder);
  EXPECT_EQ(buffer.push_imu(imu(0.9)), Result::OutOfOrder);
  EXPECT_EQ(buffer.statistics().imu_pending, 1U);
}

TEST(Ros2Buffer, FramesWaitForImuInTheCorrectClock) {
  Ros2MeasurementBuffer buffer(8, 2, 1, 100);
  ASSERT_EQ(buffer.push_camera(camera(2)), Result::Accepted);
  buffer.push_imu(imu(2.01));
  Ros2MeasurementBuffer::Batch batch;
  ASSERT_TRUE(buffer.wait_take(batch, -1, 0.02));
  EXPECT_FALSE(batch.camera);
  EXPECT_EQ(batch.imu.size(), 1U);
  buffer.push_imu(imu(2.02));
  ASSERT_TRUE(buffer.wait_take(batch, 2.01, 0.02));
  ASSERT_TRUE(batch.camera);
  EXPECT_DOUBLE_EQ(batch.camera->timestamp, 2);
}

TEST(Ros2Buffer, ReleasesOnlyTheImuThatReachesTheNextFrame) {
  Ros2MeasurementBuffer buffer(16, 4, 1, 100);
  for (double time : {1.00, 1.01, 1.02, 1.03, 1.04})
    buffer.push_imu(imu(time));
  buffer.push_camera(camera(1.015));
  Ros2MeasurementBuffer::Batch batch;
  ASSERT_TRUE(buffer.wait_take(batch, -1, 0));
  ASSERT_TRUE(batch.camera);
  ASSERT_EQ(batch.imu.size(), 3U);
  EXPECT_DOUBLE_EQ(batch.imu.back().timestamp, 1.02);
  ASSERT_TRUE(buffer.wait_take(batch, 1.02, 0));
  EXPECT_FALSE(batch.camera);
  EXPECT_EQ(batch.imu.size(), 2U);
}

TEST(Ros2Buffer, SortsIndependentCameraStreams) {
  Ros2MeasurementBuffer buffer(8, 4, 2, 100);
  buffer.push_camera(camera(2, 0));
  buffer.push_camera(camera(1, 1));
  buffer.push_imu(imu(3));
  Ros2MeasurementBuffer::Batch batch;
  ASSERT_TRUE(buffer.wait_take(batch, -1, 0));
  ASSERT_TRUE(batch.camera);
  EXPECT_DOUBLE_EQ(batch.camera->timestamp, 1);
  ASSERT_TRUE(buffer.wait_take(batch, 3, 0));
  ASSERT_TRUE(batch.camera);
  EXPECT_DOUBLE_EQ(batch.camera->timestamp, 2);
}

TEST(Ros2Buffer, BoundedCamerasKeepRecentFrames) {
  Ros2MeasurementBuffer buffer(8, 2, 1, 100);
  buffer.push_camera(camera(1));
  buffer.push_camera(camera(2));
  EXPECT_EQ(buffer.push_camera(camera(3)), Result::DroppedOldest);
  const auto stats = buffer.statistics();
  EXPECT_EQ(stats.cameras_pending, 2U);
  EXPECT_EQ(stats.camera_peak, 2U);
  EXPECT_EQ(stats.cameras_dropped, 1U);
  EXPECT_EQ(stats.resets, 1U);
  buffer.push_imu(imu(4));
  Ros2MeasurementBuffer::Batch batch;
  ASSERT_TRUE(buffer.wait_take(batch, -1, 0));
  ASSERT_TRUE(batch.camera);
  EXPECT_TRUE(batch.reset);
  EXPECT_DOUBLE_EQ(batch.camera->timestamp, 2);
}

TEST(Ros2Buffer, LongVisualGapStartsANewEpochWithFreshImu) {
  Ros2MeasurementBuffer buffer(8, 3, 1, 20, true, 0.5);
  buffer.push_camera(camera(1));
  buffer.push_imu(imu(1.01));
  Ros2MeasurementBuffer::Batch batch;
  ASSERT_TRUE(buffer.wait_take(batch, -1, 0));
  ASSERT_TRUE(batch.camera);
  EXPECT_FALSE(batch.reset);
  buffer.push_camera(camera(3));
  // A reset must not use IMU coverage belonging to the previous estimator epoch.
  ASSERT_TRUE(buffer.wait_take(batch, 4, 0));
  EXPECT_TRUE(batch.reset);
  EXPECT_FALSE(batch.camera);
  buffer.push_imu(imu(3.01));
  ASSERT_TRUE(buffer.wait_take(batch, -1, 0));
  ASSERT_TRUE(batch.camera);
  ASSERT_EQ(batch.imu.size(), 1U);
  EXPECT_DOUBLE_EQ(batch.camera->timestamp, 3);
  EXPECT_FALSE(batch.reset);
  EXPECT_EQ(buffer.statistics().resets, 1U);
}

TEST(Ros2Buffer, NativeShortCameraGapsDoNotReset) {
  Ros2MeasurementBuffer buffer(8, 3, 1, 20, true, 0.5);
  Ros2MeasurementBuffer::Batch batch;
  for (double time : {1.0, 1.1, 1.6}) {
    buffer.push_camera(camera(time));
    buffer.push_imu(imu(time + 0.01));
    ASSERT_TRUE(buffer.wait_take(batch, -1, 0));
    ASSERT_TRUE(batch.camera);
    EXPECT_FALSE(batch.reset);
  }
  EXPECT_EQ(buffer.statistics().resets, 0U);
}

TEST(Ros2Buffer, CameraOverflowDoesNotReuseOldImuCoverage) {
  Ros2MeasurementBuffer buffer(8, 1, 1, 20);
  buffer.push_camera(camera(1));
  buffer.push_camera(camera(1.05));
  Ros2MeasurementBuffer::Batch batch;
  ASSERT_TRUE(buffer.wait_take(batch, 2, 0));
  EXPECT_TRUE(batch.reset);
  EXPECT_FALSE(batch.camera);
  buffer.push_imu(imu(1.06));
  ASSERT_TRUE(buffer.wait_take(batch, -1, 0));
  ASSERT_TRUE(batch.camera);
  EXPECT_FALSE(batch.reset);
  EXPECT_EQ(buffer.statistics().resets, 1U);
  EXPECT_EQ(buffer.statistics().cameras_dropped, 1U);
}

TEST(Ros2Buffer, ImuLossRequestsEstimatorReset) {
  Ros2MeasurementBuffer buffer(2, 2, 1, 20);
  buffer.push_camera(camera(1));
  buffer.push_imu(imu(1));
  buffer.push_imu(imu(2));
  EXPECT_EQ(buffer.push_imu(imu(3)), Result::Reset);
  Ros2MeasurementBuffer::Batch batch;
  ASSERT_TRUE(buffer.wait_take(batch, -1, 0));
  EXPECT_TRUE(batch.reset);
  EXPECT_FALSE(batch.camera);
  ASSERT_EQ(batch.imu.size(), 2U);
  EXPECT_DOUBLE_EQ(batch.imu.front().timestamp, 2);
  EXPECT_EQ(buffer.statistics().imu_dropped, 1U);
}

TEST(Ros2Buffer, RewindStartsANewEpoch) {
  Ros2MeasurementBuffer buffer(8, 2, 1, 20);
  buffer.push_imu(imu(10));
  buffer.push_camera(camera(10));
  EXPECT_EQ(buffer.push_imu(imu(1)), Result::Reset);
  EXPECT_EQ(buffer.push_camera(camera(1)), Result::Accepted);
  buffer.push_imu(imu(1.01));
  Ros2MeasurementBuffer::Batch batch;
  ASSERT_TRUE(buffer.wait_take(batch, 10, 0));
  ASSERT_TRUE(batch.reset);
  ASSERT_TRUE(batch.camera);
  EXPECT_DOUBLE_EQ(batch.camera->timestamp, 1);
  EXPECT_EQ(buffer.statistics().resets, 1U);
}

TEST(Ros2Buffer, RateLimitToleratesTimestampRounding) {
  Ros2MeasurementBuffer buffer(8, 4, 1, 20);
  const double epoch = 1400000000;
  EXPECT_EQ(buffer.push_camera(camera(epoch)), Result::Accepted);
  EXPECT_EQ(buffer.push_camera(camera(epoch + 0.05)), Result::Accepted);
  EXPECT_EQ(buffer.push_camera(camera(epoch + 0.051)), Result::RateLimited);
}

TEST(Ros2Buffer, InvalidCameraDoesNotPoisonTimestamp) {
  Ros2MeasurementBuffer buffer(8, 4, 1, 20);
  auto invalid = camera(5);
  invalid.masks[0] = cv::Mat::zeros(1, 1, CV_8UC1);
  EXPECT_EQ(buffer.push_camera(invalid), Result::Invalid);
  EXPECT_EQ(buffer.push_camera(camera(4)), Result::Accepted);
}

TEST(Ros2Buffer, StopUnblocksAnIdleConsumer) {
  Ros2MeasurementBuffer buffer(8, 4, 1, 20);
  auto consumer = std::async(std::launch::async, [&] {
    Ros2MeasurementBuffer::Batch batch;
    return buffer.wait_take(batch, -1, 0);
  });
  EXPECT_EQ(consumer.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
  buffer.stop();
  EXPECT_EQ(consumer.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  EXPECT_FALSE(consumer.get());
  EXPECT_EQ(buffer.push_imu(imu(1)), Result::Stopped);
}

TEST(Ros2Buffer, ConcurrentProducerConsumerStayBounded) {
  Ros2MeasurementBuffer buffer(32, 4, 1, 1000);
  std::thread consumer([&] {
    double fed_time = -1;
    Ros2MeasurementBuffer::Batch batch;
    while (buffer.wait_take(batch, fed_time, 0)) {
      if (batch.reset)
        fed_time = -1;
      for (const auto &sample : batch.imu)
        fed_time = sample.timestamp;
    }
  });
  for (int i = 1; i <= 10000; ++i) {
    buffer.push_imu(imu(i * 0.001));
    if (i % 5 == 0)
      buffer.push_camera(camera(i * 0.001));
  }
  buffer.stop();
  consumer.join();
  const auto stats = buffer.statistics();
  EXPECT_LE(stats.imu_peak, 32U);
  EXPECT_LE(stats.camera_peak, 4U);
  EXPECT_EQ(stats.imu_accepted, 10000U);
}
