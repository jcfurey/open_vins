/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <gtest/gtest.h>

#include "cam/CamRadtan.h"
#include "track/TrackKLT.h"

namespace {
class Tracker : public ov_core::TrackKLT {
public:
  using TrackKLT::TrackKLT;

  void discard_scratch() {
    for (auto &buffer : img_curr)
      buffer.second.release();
    for (auto &pyramid : img_pyramid_curr)
      pyramid.second.clear();
    for (auto &filter : clahe)
      filter.second = cv::createCLAHE(10.0, cv::Size(8, 8));
  }

  cv::Mat last_image(size_t camera) const { return img_last.at(camera); }
};

std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cameras(int count, int width = 192, int height = 128) {
  std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> result;
  for (int i = 0; i < count; ++i) {
    auto camera = std::make_shared<ov_core::CamRadtan>(width, height);
    Eigen::Matrix<double, 8, 1> intrinsics;
    intrinsics << .625 * width, .9375 * height, width / 2.0, height / 2.0, 0, 0, 0, 0;
    camera->set_value(intrinsics);
    result[i] = camera;
  }
  return result;
}

void compare_tracks(bool stereo, int camera_count, ov_core::TrackBase::HistogramMethod method, int levels = 5, int window = 15,
                    int width = 192, int height = 128, int threads = 1) {
  cv::setNumThreads(threads);
  Tracker reused(cameras(camera_count, width, height), 100, 0, stereo, method, 10, 3, 2, 6, levels, window);
  Tracker fresh(cameras(camera_count, width, height), 100, 0, stereo, method, 10, 3, 2, 6, levels, window);
  cv::Mat texture(height + 32, width + 64, CV_8UC1);
  cv::RNG random(42);
  random.fill(texture, cv::RNG::UNIFORM, 0, 256);
  size_t tracked = 0;
  for (int frame = 0; frame < 16; ++frame) {
    ov_core::CameraData data;
    data.timestamp = 1.0 + frame * .05;
    for (int camera = 0; camera < camera_count; ++camera) {
      data.sensor_ids.push_back(camera);
      // Include blank frames to exercise failed matching and redetection.
      data.images.push_back(frame == 6 || frame == 7 ? cv::Mat::zeros(height, width, CV_8UC1)
                                                   : texture(cv::Rect(10 + frame + camera * 3, 8, width, height)).clone());
      data.masks.push_back(cv::Mat::zeros(height, width, CV_8UC1));
    }
    fresh.discard_scratch(); // Reference: recreate preprocessing storage and CLAHE on every frame.
    cv::setRNGSeed(frame);
    reused.feed_new_camera(data);
    cv::setRNGSeed(frame);
    fresh.feed_new_camera(data);
    EXPECT_EQ(reused.get_last_ids(), fresh.get_last_ids()) << "frame " << frame;
    auto actual = reused.get_last_obs(), expected = fresh.get_last_obs();
    for (int camera = 0; camera < camera_count; ++camera) {
      EXPECT_EQ(cv::norm(reused.last_image(camera), fresh.last_image(camera), cv::NORM_INF), 0);
      ASSERT_EQ(actual.at(camera).size(), expected.at(camera).size());
      tracked += actual.at(camera).size();
      for (size_t point = 0; point < actual.at(camera).size(); ++point)
        EXPECT_EQ(actual.at(camera)[point].pt, expected.at(camera)[point].pt) << "frame " << frame;
    }
  }
  EXPECT_GT(tracked, 100u); // Avoid vacuously comparing two failed trackers.
}
} // namespace

TEST(KltBuffers, StereoMatchesFreshPyramidsAcrossResets) {
  for (auto method : {ov_core::TrackBase::NONE, ov_core::TrackBase::HISTOGRAM, ov_core::TrackBase::CLAHE})
    compare_tracks(true, 2, method);
}

TEST(KltBuffers, MonocularMatchesFreshPyramidsAcrossResets) {
  compare_tracks(false, 1, ov_core::TrackBase::HISTOGRAM);
}

TEST(KltBuffers, BinocularMatchesFreshPyramidsAcrossResets) {
  compare_tracks(false, 3, ov_core::TrackBase::HISTOGRAM);
}

TEST(KltBuffers, ConfigurablePyramidsAndWindowsPreserveBufferIsolation) {
  compare_tracks(true, 2, ov_core::TrackBase::HISTOGRAM, 3, 21);
  compare_tracks(false, 3, ov_core::TrackBase::HISTOGRAM, 2, 9);
}

TEST(KltBuffers, EurocResolutionWithTwoOpenCvWorkersMatchesFreshPyramids) {
  compare_tracks(true, 2, ov_core::TrackBase::HISTOGRAM, 5, 15, 752, 480, 2);
}

TEST(KltBuffers, InvalidFlowSettingsAreRejectedBeforeTracking) {
  for (auto setting : {std::make_pair(-1, 15), std::make_pair(9, 15), std::make_pair(3, 2),
                       std::make_pair(3, 4), std::make_pair(3, 65)}) {
    EXPECT_THROW((Tracker(cameras(2), 100, 0, true, ov_core::TrackBase::NONE, 10, 3, 2, 6, setting.first, setting.second)),
                 std::invalid_argument);
  }
}
