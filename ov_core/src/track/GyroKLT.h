/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef OV_CORE_GYRO_KLT_H
#define OV_CORE_GYRO_KLT_H

#include "cam/CamBase.h"
#include <Eigen/Dense>
#include <map>
#include <opencv2/video/tracking.hpp>
#include <unordered_map>

namespace ov_core {

/// Per-camera, optional motion assistance. Gyro is a search/patch-shape aid;
/// final observations stay in the original, distorted image coordinates.
class GyroKLT {
public:
  struct Statistics {
    size_t predictions = 0, refinements = 0, references = 0, rejected = 0, missing_frames = 0;
  };
  GyroKLT(std::shared_ptr<CamBase> camera, std::string mode, int window, int reference_age = 20);
  void begin_frame(const Eigen::Matrix3d *previous_to_current);
  void seed(const std::vector<cv::Point2f> &previous, std::vector<cv::Point2f> &current);
  void refine(const std::vector<cv::Mat> &previous_raw, const std::vector<cv::Mat> &current_raw,
              const std::vector<cv::Point2f> &previous, const std::vector<size_t> &ids,
              std::vector<cv::Point2f> &current, std::vector<uchar> &status, std::vector<float> &error);
  void rekey(size_t previous, size_t current);
  void retain(const std::vector<size_t> &ids);
  bool valid() const { return prediction_valid; }
  bool uses_patches() const { return mode != "seed"; }
  Statistics statistics() const { return stats; }

  /// Distortion-aware rotational point projection and its local affine derivative.
  static bool project(CamBase &camera, const Eigen::Matrix3d &rotation, const cv::Point2f &point,
                      cv::Point2f &predicted, Eigen::Matrix2d *affine = nullptr);

private:
  struct Reference {
    cv::Point2f point;
    Eigen::Matrix3d orientation;
    std::vector<cv::Mat> patches; // Owned float templates, never borrowed double-buffer views.
    size_t frame = 0;
  };
  Reference make_reference(const std::vector<cv::Mat> &pyramid, const cv::Point2f &point,
                           const Eigen::Matrix3d &orientation) const;
  bool align(const Reference &reference, const std::vector<cv::Mat> &current, const Eigen::Matrix2d &affine,
             cv::Point2f &point, float &error) const;
  std::shared_ptr<CamBase> camera;
  std::string mode;
  int window, reference_age;
  size_t frame = 0;
  bool prediction_valid = false;
  Eigen::Matrix3d step = Eigen::Matrix3d::Identity(), previous_orientation = Eigen::Matrix3d::Identity(),
                  orientation = Eigen::Matrix3d::Identity();
  std::unordered_map<size_t, Reference> references;
  Statistics stats;
};
} // namespace ov_core
#endif
