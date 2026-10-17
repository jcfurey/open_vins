/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <gtest/gtest.h>

#include "feat/FeatureDatabase.h"
#include "init/InertialInitializer.h"
#include "types/IMU.h"

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
