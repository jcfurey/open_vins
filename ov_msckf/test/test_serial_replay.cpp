/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>

#include "cam/CamRadtan.h"
#include "core/VioManager.h"
#include "replay/AslDataset.h"
#include "replay/FrameSynchronizer.h"
#include "replay/SerialFeeder.h"
#include "replay/TrajectoryWriter.h"
#include "state/State.h"

using namespace ov_msckf;
namespace fs = boost::filesystem;

namespace {

VioManagerOptions mono_options(double camera_to_imu_dt) {
  VioManagerOptions options;
  options.state_options.num_cameras = 1;
  options.init_options.num_cameras = 1;
  options.use_aruco = false;
  options.vec_dw << 1, 0, 0, 1, 0, 1;
  options.vec_da = options.vec_dw;
  options.vec_tg.setZero();
  options.q_ACCtoIMU << 0, 0, 0, 1;
  options.q_GYROtoIMU = options.q_ACCtoIMU;
  options.calib_camimu_dt = camera_to_imu_dt;
  options.track_frequency = 20.0;
  auto camera = std::make_shared<ov_core::CamRadtan>(64, 48);
  Eigen::Matrix<double, 8, 1> calibration;
  calibration << 50, 50, 32, 24, 0, 0, 0, 0;
  camera->set_value(calibration);
  options.camera_intrinsics[0] = camera;
  Eigen::VectorXd extrinsic(7);
  extrinsic << 0, 0, 0, 1, 0, 0, 0;
  options.camera_extrinsics[0] = extrinsic;
  options.init_options.camera_intrinsics = options.camera_intrinsics;
  options.init_options.camera_extrinsics = options.camera_extrinsics;
  return options;
}

ov_core::ImuData imu(double time) { return {time, Eigen::Vector3d::Zero(), Eigen::Vector3d(0, 0, 9.81)}; }

ov_core::CameraData frame(double time) {
  ov_core::CameraData packet;
  packet.timestamp = time;
  packet.sensor_ids = {0};
  packet.images = {cv::Mat::zeros(48, 64, CV_8UC1)};
  packet.masks = {cv::Mat::zeros(48, 64, CV_8UC1)};
  return packet;
}

class InitializedManager : public VioManager {
public:
  InitializedManager(VioManagerOptions &options, const Eigen::Matrix<double, 17, 1> &truth) : VioManager(options) {
    initialize_with_gt(truth);
    timelastupdate = truth(0); // as if one update had completed
  }
};

struct TempDir {
  fs::path path = fs::temp_directory_path() / fs::unique_path("ov_replay_%%%%-%%%%");
  TempDir() { fs::create_directories(path); }
  ~TempDir() { fs::remove_all(path); }
};

void write(const fs::path &file, const std::string &text) {
  fs::create_directories(file.parent_path());
  std::ofstream(file.string()) << text;
}

} // namespace

TEST(AslSeconds, MatchesRosStampConversion) {
  const int64_t ns = 1403636579758555392LL;
  EXPECT_EQ(asl_seconds(ns), 1403636579.0 + 758555392 * 1e-9);
  EXPECT_EQ(asl_seconds(999999999), 999999999 * 1e-9);
}

TEST(AslDataset, ParsesImuAndGroupsStereoFrames) {
  TempDir dir;
  const fs::path mav0 = dir.path / "mav0";
  write(mav0 / "imu0" / "data.csv", "#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\r\n"
                                    "1000000000,0.1,0.2,0.3,1,2,3\r\n"
                                    "1005000000,0.1,0.2,0.3,1,2,3.5\r\n"
                                    "1005000000,9,9,9,9,9,9\r\n");
  write(mav0 / "cam0" / "data.csv", "#timestamp [ns],filename\n1000000000,a.png\n1050000000,b.png\n1100000000,c.png\n");
  write(mav0 / "cam1" / "data.csv", "#timestamp [ns],filename\n1000000000,a.png\n1100000300,c.png\n");

  const AslDataset dataset(dir.path.string(), 2, 0.001);
  ASSERT_EQ(dataset.imu().size(), 2U); // the duplicate stamp is dropped
  EXPECT_DOUBLE_EQ(dataset.imu().back().am.z(), 3.5);
  ASSERT_EQ(dataset.frames().size(), 2U);
  EXPECT_EQ(dataset.frames()[1].timestamp_ns, 1100000000);
  EXPECT_EQ(dataset.frames()[1].paths[1], (mav0 / "cam1" / "data" / "c.png").string());
  EXPECT_EQ(dataset.unmatched_frames(), 1U);

  // The mav0 folder itself is accepted too
  EXPECT_EQ(AslDataset(mav0.string(), 1).frames().size(), 3U);
}

TEST(AslDataset, RejectsMalformedRows) {
  TempDir dir;
  write(dir.path / "imu0" / "data.csv", "1000000000,0.1,0.2,0.3,1,2\n");
  write(dir.path / "cam0" / "data.csv", "1000000000,a.png\n");
  EXPECT_THROW(AslDataset(dir.path.string(), 1), std::runtime_error);
}

TEST(ImagePrefetcher, DeliversFramesInOrderAndReportsDecodeErrors) {
  TempDir dir;
  std::vector<AslDataset::Frame> frames;
  for (int i = 0; i < 5; i++) {
    const std::string path = (dir.path / (std::to_string(i) + ".png")).string();
    cv::imwrite(path, cv::Mat(4, 6, CV_8UC1, cv::Scalar(10 * i)));
    frames.push_back({i, {path}});
  }
  frames.push_back({5, {(dir.path / "missing.png").string()}});
  ImagePrefetcher prefetcher(frames, 2);
  int64_t stamp;
  std::vector<cv::Mat> images;
  for (int i = 0; i < 5; i++) {
    ASSERT_TRUE(prefetcher.next(stamp, images));
    EXPECT_EQ(stamp, i);
    EXPECT_EQ(images.at(0).at<uint8_t>(0, 0), 10 * i);
  }
  EXPECT_THROW(prefetcher.next(stamp, images), std::runtime_error);
}

TEST(SerialFeeder, ReleasesFramesOnlyOnceImuCoversTheOffsetTime) {
  auto options = mono_options(0.01);
  auto app = std::make_shared<VioManager>(options);
  std::vector<double> processed;
  SerialFeeder feeder(app, [&](const ov_core::CameraData &packet, double) { processed.push_back(packet.timestamp); });

  EXPECT_TRUE(feeder.feed_imu(imu(0.995)));
  EXPECT_TRUE(feeder.feed_camera(frame(1.0)));
  EXPECT_TRUE(processed.empty());
  EXPECT_TRUE(feeder.feed_imu(imu(1.005)));
  EXPECT_TRUE(processed.empty()); // 1.005 < 1.0 + 0.01
  EXPECT_TRUE(feeder.feed_imu(imu(1.010)));
  ASSERT_EQ(processed.size(), 1U);
  EXPECT_DOUBLE_EQ(processed[0], 1.0);
}

TEST(SerialFeeder, RejectsStaleRateLimitedAndInvalidInput) {
  auto options = mono_options(0.0);
  auto app = std::make_shared<VioManager>(options);
  SerialFeeder feeder(app);
  EXPECT_TRUE(feeder.feed_imu(imu(1.0)));
  EXPECT_FALSE(feeder.feed_imu(imu(1.0)));
  EXPECT_FALSE(feeder.feed_imu(imu(std::nan(""))));
  EXPECT_TRUE(feeder.feed_camera(frame(2.0)));
  EXPECT_FALSE(feeder.feed_camera(frame(2.0)));  // not increasing
  EXPECT_FALSE(feeder.feed_camera(frame(2.02))); // faster than track_frequency
  EXPECT_TRUE(feeder.feed_camera(frame(2.05)));  // exactly at the rate is accepted
  auto bad = frame(3.0);
  bad.masks[0] = cv::Mat::zeros(10, 10, CV_8UC1);
  EXPECT_FALSE(feeder.feed_camera(bad));
  feeder.finish();
  const auto &stats = feeder.statistics();
  EXPECT_EQ(stats.imu_accepted, 1U);
  EXPECT_EQ(stats.imu_rejected, 2U);
  EXPECT_EQ(stats.cameras_accepted, 2U);
  EXPECT_EQ(stats.cameras_rate_limited, 1U);
  EXPECT_EQ(stats.cameras_rejected, 2U);
  EXPECT_EQ(stats.cameras_processed, 0U);
  EXPECT_EQ(stats.cameras_unprocessed, 2U); // IMU never reached the frames
}

TEST(TrajectoryWriter, WritesNothingBeforeInitialization) {
  TempDir dir;
  auto options = mono_options(0.0);
  VioManager app(options);
  TrajectoryWriter writer((dir.path / "out" / "trajectory.txt").string());
  EXPECT_FALSE(writer.write(app));
  EXPECT_EQ(writer.count(), 0U);
  EXPECT_EQ(writer.digest(), TrajectoryWriter((dir.path / "other.txt").string()).digest());
}

TEST(TrajectoryWriter, WritesImuClockPoseWithCovariances) {
  TempDir dir;
  auto options = mono_options(0.25);
  Eigen::Matrix<double, 17, 1> truth = Eigen::Matrix<double, 17, 1>::Zero();
  truth(0) = 10;
  truth(4) = 1;
  truth.segment<3>(5) << 1, 2, 3;
  InitializedManager app(options, truth);
  const std::string path = (dir.path / "trajectory.txt").string();
  std::string digest;
  {
    TrajectoryWriter writer(path);
    EXPECT_TRUE(writer.write(app));
    EXPECT_FALSE(writer.write(app)); // same state time
    app.get_state()->_timestamp = 10.05;
    EXPECT_TRUE(writer.write(app));
    EXPECT_EQ(writer.count(), 2U);
    EXPECT_DOUBLE_EQ(writer.first_time(), 10.25);
    EXPECT_DOUBLE_EQ(writer.last_time(), 10.3);
    digest = writer.digest();
  }
  std::ifstream input(path);
  std::string line;
  std::getline(input, line);
  EXPECT_EQ(line.rfind("# timestamp(s) tx ty tz qx qy qz qw", 0), 0U);
  std::getline(input, line);
  std::istringstream fields(line);
  std::vector<double> values;
  for (double value; fields >> value;)
    values.push_back(value);
  ASSERT_EQ(values.size(), 20U);
  EXPECT_DOUBLE_EQ(values[0], 10.25);
  EXPECT_DOUBLE_EQ(values[1], 1);
  EXPECT_DOUBLE_EQ(values[3], 3);
  EXPECT_DOUBLE_EQ(values[7], 1);               // qw
  EXPECT_NEAR(values[8], 0.017 * 0.017, 1e-12); // Pr11
  EXPECT_NEAR(values[9], 0.0, 1e-12);           // Pr12
  EXPECT_NEAR(values[14], 0.05 * 0.05, 1e-12);  // Pt11
  EXPECT_NEAR(values[19], 0.05 * 0.05, 1e-12);  // Pt33

  // Identical writes give identical digests; any bit change alters it.
  InitializedManager same(options, truth);
  TrajectoryWriter again((dir.path / "again.txt").string());
  again.write(same);
  same.get_state()->_timestamp = 10.05;
  again.write(same);
  EXPECT_EQ(again.digest(), digest);
}

TEST(FrameSynchronizer, PairsWithinToleranceAndDropsOrphans) {
  FrameSynchronizer sync(2, 0.005);
  FrameSynchronizer::Frame frame;
  auto image = [](int value) { return cv::Mat(1, 1, CV_8UC1, cv::Scalar(value)); };
  EXPECT_TRUE(sync.push(0, 1.00, image(1)));
  EXPECT_FALSE(sync.pop(frame)); // waits for camera 1
  EXPECT_TRUE(sync.push(1, 1.002, image(2)));
  ASSERT_TRUE(sync.pop(frame));
  EXPECT_DOUBLE_EQ(frame.timestamp, 1.00);
  EXPECT_EQ(frame.images.at(1).at<uint8_t>(0, 0), 2);
  EXPECT_TRUE(sync.push(0, 1.05, image(3))); // never gets a partner
  EXPECT_TRUE(sync.push(1, 1.10, image(4)));
  EXPECT_FALSE(sync.pop(frame));
  EXPECT_TRUE(sync.push(0, 1.10, image(5)));
  ASSERT_TRUE(sync.pop(frame));
  EXPECT_DOUBLE_EQ(frame.timestamp, 1.10);
  EXPECT_EQ(sync.unmatched(), 1U);
  EXPECT_FALSE(sync.push(0, 1.10, image(6))); // stale even though its queue is empty
  EXPECT_FALSE(sync.push(2, 1.20, image(7))); // unknown camera
  EXPECT_TRUE(sync.push(1, 1.30, image(8)));
  EXPECT_EQ(sync.unmatched(), 2U); // still pending at end of input
}
