/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <gtest/gtest.h>

#include "feat/FeatureDatabase.h"
#include "init/InertialInitializer.h"
#include "types/IMU.h"
#include "utils/helper.h"

#include <limits>

class TestInitializer : public ov_init::InertialInitializer {
public:
  using InertialInitializer::InertialInitializer;
  const std::vector<ov_core::ImuData> &samples() const { return *imu_data; }
};

namespace {
ov_core::ImuData sample(double time) {
  return {time, Eigen::Vector3d(0.01, -0.02, 0.03), Eigen::Vector3d(0, 0, 9.81)};
}

struct Startup {
  ov_init::InertialInitializerOptions options;
  std::shared_ptr<ov_core::FeatureDatabase> features = std::make_shared<ov_core::FeatureDatabase>();
  Startup(bool moving = false) {
    options.init_window_time = 2;
    options.init_imu_thresh = 1.5;
    options.init_max_disparity = 10;
    options.init_dyn_use = false;
    for (int frame = 0; frame <= 42; ++frame) {
      const double time = 10 + frame * 0.05;
      const float motion = moving && time > 11.1 ? (time - 11.1) * 100 : 0;
      for (int id = 0; id < 25; ++id)
        features->update_feature(id, time, 0, id * 5 + motion, 50, 0, 0);
    }
  }
  void feed(TestInitializer &initializer, bool moving = false) {
    for (int i = 0; i <= 420; ++i) {
      auto measurement = sample(10 + i * 0.005);
      if (moving && measurement.timestamp > 11.1)
        measurement.am.x() = i % 2 ? 3 : -3;
      initializer.feed_imu(measurement);
    }
  }
};
} // namespace

TEST(InitializationHistory, RetainsCutoffAndCompactsAllOlderSamples) {
  Startup startup;
  TestInitializer initializer(startup.options, startup.features);
  for (double time : {1.0, 3.0, 2.0, 4.0})
    initializer.feed_imu(sample(time));
  initializer.feed_imu(sample(5), 3);
  ASSERT_EQ(initializer.samples().size(), 3U);
  EXPECT_DOUBLE_EQ(initializer.samples()[0].timestamp, 3);
  EXPECT_DOUBLE_EQ(initializer.samples()[1].timestamp, 4);
  EXPECT_DOUBLE_EQ(initializer.samples()[2].timestamp, 5);
}

TEST(InitializationHistory, CanDiscardTheEntirePreviousWindow) {
  Startup startup;
  TestInitializer initializer(startup.options, startup.features);
  initializer.feed_imu(sample(1));
  initializer.feed_imu(sample(10), 9);
  ASSERT_EQ(initializer.samples().size(), 1U);
  EXPECT_DOUBLE_EQ(initializer.samples().front().timestamp, 10);
}

TEST(InitializationOptions, RejectsInsufficientOrUnattainableDynamicFeatureSupport) {
  ov_init::InertialInitializerOptions options;
  options.init_dyn_min_features = 7;
  EXPECT_THROW(options.print_and_load_initializer(), std::invalid_argument);
  options.init_dyn_min_features = options.init_max_features + 1;
  EXPECT_THROW(options.print_and_load_initializer(), std::invalid_argument);
  options.init_dyn_min_features = 25;
  EXPECT_NO_THROW(options.print_and_load_initializer());
}

TEST(InitializationImuInterval, ExactLastSampleClosesTheRequestedInterval) {
  const std::vector<ov_core::ImuData> data = {sample(0), sample(0.005), sample(0.010)};
  const auto selected = ov_init::InitializerHelper::select_imu_readings(data, 0, 0.010);
  ASSERT_EQ(selected.size(), 3U);
  EXPECT_DOUBLE_EQ(selected.front().timestamp, 0);
  EXPECT_DOUBLE_EQ(selected.back().timestamp, 0.010);
}

TEST(InitializationImuInterval, FutureInputDoesNotChangeTheSelectedInterval) {
  std::vector<ov_core::ImuData> data = {sample(0), sample(0.005), sample(0.010)};
  const auto before = ov_init::InitializerHelper::select_imu_readings(data, 0, 0.010);
  data.push_back(sample(0.015));
  const auto after = ov_init::InitializerHelper::select_imu_readings(data, 0, 0.010);
  ASSERT_EQ(before.size(), after.size());
  for (size_t i = 0; i < before.size(); ++i) {
    EXPECT_DOUBLE_EQ(before[i].timestamp, after[i].timestamp);
    EXPECT_EQ((before[i].wm - after[i].wm).norm(), 0);
    EXPECT_EQ((before[i].am - after[i].am).norm(), 0);
  }
}

TEST(InitializationImuInterval, InterpolatesBothBoundaries) {
  std::vector<ov_core::ImuData> data;
  for (double time : {0.0, 0.5, 1.0, 1.5}) {
    auto measurement = sample(time);
    measurement.wm = time * Eigen::Vector3d(1, 2, 3);
    measurement.am = time * Eigen::Vector3d(4, 5, 6);
    data.push_back(measurement);
  }
  const auto selected = ov_init::InitializerHelper::select_imu_readings(data, 0.25, 1.25);
  ASSERT_EQ(selected.size(), 4U);
  EXPECT_DOUBLE_EQ(selected.front().timestamp, 0.25);
  EXPECT_DOUBLE_EQ(selected.back().timestamp, 1.25);
  for (const auto &measurement : selected) {
    EXPECT_NEAR((measurement.wm - measurement.timestamp * Eigen::Vector3d(1, 2, 3)).norm(), 0, 1e-12);
    EXPECT_NEAR((measurement.am - measurement.timestamp * Eigen::Vector3d(4, 5, 6)).norm(), 0, 1e-12);
  }
}

TEST(InitializationImuInterval, TwoSampleBufferRetainsBothBounds) {
  const auto selected = ov_init::InitializerHelper::select_imu_readings({sample(0), sample(0.005)}, 0, 0.005);
  ASSERT_EQ(selected.size(), 2U);
  EXPECT_DOUBLE_EQ(selected.front().timestamp, 0);
  EXPECT_DOUBLE_EQ(selected.back().timestamp, 0.005);
}

TEST(InitializationImuInterval, RejectsIntervalsWithoutBothBrackets) {
  const std::vector<ov_core::ImuData> data = {sample(0), sample(0.5), sample(1)};
  EXPECT_TRUE(ov_init::InitializerHelper::select_imu_readings(data, -0.1, 0.5).empty());
  EXPECT_TRUE(ov_init::InitializerHelper::select_imu_readings(data, 0, 1.1).empty());
  EXPECT_TRUE(ov_init::InitializerHelper::select_imu_readings({}, 0, 1).empty());
  EXPECT_TRUE(ov_init::InitializerHelper::select_imu_readings({sample(0)}, 0, 1).empty());
}

TEST(InitializationImuInterval, RejectsInvalidRequestedTimes) {
  const std::vector<ov_core::ImuData> data = {sample(0), sample(0.5), sample(1)};
  EXPECT_TRUE(ov_init::InitializerHelper::select_imu_readings(data, 0, 0).empty());
  EXPECT_TRUE(ov_init::InitializerHelper::select_imu_readings(data, 0.5, 0).empty());
  EXPECT_TRUE(ov_init::InitializerHelper::select_imu_readings(data, 0, std::numeric_limits<double>::infinity()).empty());
  EXPECT_TRUE(ov_init::InitializerHelper::select_imu_readings(data, std::numeric_limits<double>::quiet_NaN(), 1).empty());
}

TEST(InitializationImuInterval, RejectsNonfiniteMeasurements) {
  std::vector<ov_core::ImuData> data = {sample(0), sample(0.5), sample(1)};
  data[1].wm.x() = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(ov_init::InitializerHelper::select_imu_readings(data, 0, 1).empty());
}

TEST(InitializationImuInterval, RejectsNonfiniteOrOutOfOrderTimestamps) {
  std::vector<ov_core::ImuData> data = {sample(0), sample(0.5), sample(1)};
  data[1].timestamp = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(ov_init::InitializerHelper::select_imu_readings(data, 0, 1).empty());
  data = {sample(0), sample(0.75), sample(0.5), sample(1)};
  EXPECT_TRUE(ov_init::InitializerHelper::select_imu_readings(data, 0, 1).empty());
}

TEST(InitializationImuInterval, NearCoincidentSamplesDoNotCreateZeroLengthSteps) {
  const std::vector<ov_core::ImuData> data = {sample(0), sample(0.5), sample(0.5 + 5e-13), sample(1)};
  const auto selected = ov_init::InitializerHelper::select_imu_readings(data, 0, 1);
  ASSERT_EQ(selected.size(), 3U);
  EXPECT_DOUBLE_EQ(selected.front().timestamp, 0);
  EXPECT_DOUBLE_EQ(selected.back().timestamp, 1);
  for (size_t i = 1; i < selected.size(); ++i)
    EXPECT_GT(selected[i].timestamp - selected[i - 1].timestamp, 1e-12);
}

TEST(DynamicInitialization, DeferredAttemptRetainsTheLeftImuBracket) {
  Startup startup(true);
  startup.options.init_dyn_use = true;
  TestInitializer initializer(startup.options, startup.features);
  startup.feed(initializer, true);
  double time = -1;
  Eigen::MatrixXd covariance;
  std::vector<std::shared_ptr<ov_type::Type>> order;
  // Twenty-five tracks trigger a deferred dynamic attempt before triangulation.
  ASSERT_FALSE(initializer.initialize(time, covariance, order, std::make_shared<ov_type::IMU>(), false));
  ASSERT_GE(initializer.samples().size(), 2U);
  const double oldest_camera_time = 10 + 42 * 0.05 - startup.options.init_window_time;
  EXPECT_LT(initializer.samples().front().timestamp, oldest_camera_time);
  EXPECT_NEAR(initializer.samples()[1].timestamp, oldest_camera_time, 1e-12);
}

TEST(DynamicInitialization, MissingRightImuBracketDefersWithoutChangingState) {
  Startup startup(true);
  startup.options.init_dyn_use = true;
  for (int frame = 0; frame <= 42; ++frame) {
    const double time = 10 + frame * 0.05;
    const float motion = time > 11.1 ? (time - 11.1) * 100 : 0;
    for (int id = 25; id < 50; ++id)
      startup.features->update_feature(id, time, 0, id * 5 + motion, 50, 0, 0);
  }
  TestInitializer initializer(startup.options, startup.features);
  for (int i = 0; i <= 380; ++i)
    initializer.feed_imu(sample(10 + i * 0.005));
  double time = -1;
  Eigen::MatrixXd covariance;
  std::vector<std::shared_ptr<ov_type::Type>> order;
  auto imu = std::make_shared<ov_type::IMU>();
  const auto before = imu->value();
  EXPECT_FALSE(initializer.initialize(time, covariance, order, imu, false));
  EXPECT_EQ((imu->value() - before).norm(), 0);
}

TEST(StaticInitialization, StationaryStartRecoversGravityAndGyroBiasWithoutWaitingForMotion) {
  Startup startup;
  TestInitializer initializer(startup.options, startup.features);
  startup.feed(initializer);
  double time = -1;
  Eigen::MatrixXd covariance;
  std::vector<std::shared_ptr<ov_type::Type>> order;
  auto imu = std::make_shared<ov_type::IMU>();
  ASSERT_TRUE(initializer.initialize(time, covariance, order, imu, false));
  EXPECT_NEAR((imu->bias_g() - sample(0).wm).norm(), 0, 1e-12);
  EXPECT_NEAR(imu->bias_a().norm(), 0, 1e-12);
  EXPECT_NEAR(imu->vel().norm(), 0, 1e-12);
  EXPECT_NEAR((imu->Rot() * Eigen::Vector3d(0, 0, 9.81) - sample(0).am).norm(), 0, 1e-12);
  EXPECT_TRUE(covariance.allFinite());
  ASSERT_EQ(order.size(), 1U);
  EXPECT_GT(time, 11);
  EXPECT_LT(time, 11.2);
}

TEST(StaticInitialization, ReturnsTheStateTimeInTheCameraClock) {
  auto initialize_with_offset = [](double offset) {
    Startup startup;
    startup.options.calib_camimu_dt = offset;
    TestInitializer initializer(startup.options, startup.features);
    startup.feed(initializer);
    double time = -1;
    Eigen::MatrixXd covariance;
    std::vector<std::shared_ptr<ov_type::Type>> order;
    EXPECT_TRUE(initializer.initialize(time, covariance, order, std::make_shared<ov_type::IMU>(), false));
    return time;
  };
  // t_imu = t_cam + calib_camimu_dt, and the IMU window is the same for both offsets
  EXPECT_DOUBLE_EQ(initialize_with_offset(0.02), initialize_with_offset(0) - 0.02);
}

TEST(StaticInitialization, ZeroSpecificForceDoesNotProduceAnInvalidOrientation) {
  Startup startup;
  TestInitializer initializer(startup.options, startup.features);
  for (int i = 0; i <= 420; ++i) {
    auto measurement = sample(10 + i * 0.005);
    measurement.am.setZero();
    initializer.feed_imu(measurement);
  }
  double time = -1;
  Eigen::MatrixXd covariance;
  std::vector<std::shared_ptr<ov_type::Type>> order;
  auto imu = std::make_shared<ov_type::IMU>();
  const auto before = imu->value();
  EXPECT_FALSE(initializer.initialize(time, covariance, order, imu, false));
  EXPECT_EQ((imu->value() - before).norm(), 0);
  EXPECT_TRUE(imu->value().allFinite());
}

TEST(StaticInitialization, DefaultPolicyWaitsForMotion) {
  Startup startup;
  TestInitializer initializer(startup.options, startup.features);
  startup.feed(initializer);
  double time = -1;
  Eigen::MatrixXd covariance;
  std::vector<std::shared_ptr<ov_type::Type>> order;
  EXPECT_FALSE(initializer.initialize(time, covariance, order, std::make_shared<ov_type::IMU>(), true));
}

TEST(StaticInitialization, MotionPolicyAcceptsAStationaryWindowFollowedByExcitation) {
  Startup startup(true);
  TestInitializer initializer(startup.options, startup.features);
  startup.feed(initializer, true);
  double time = -1;
  Eigen::MatrixXd covariance;
  std::vector<std::shared_ptr<ov_type::Type>> order;
  ASSERT_TRUE(initializer.initialize(time, covariance, order, std::make_shared<ov_type::IMU>(), true));
}

TEST(StaticInitialization, ImmediatePolicyRejectsMovingVisualTracks) {
  Startup startup(true);
  TestInitializer initializer(startup.options, startup.features);
  startup.feed(initializer, true);
  double time = -1;
  Eigen::MatrixXd covariance;
  std::vector<std::shared_ptr<ov_type::Type>> order;
  EXPECT_FALSE(initializer.initialize(time, covariance, order, std::make_shared<ov_type::IMU>(), false));
}
