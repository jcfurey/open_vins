/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_CORE_TRACK_KLT_H
#define OV_CORE_TRACK_KLT_H

#include "TrackBase.h"
#include "TrackQuality.h"
#include "NeuralFlow.h"
#include "LearnedFeatures.h"
#include <cmath>
#include <stdexcept>

namespace ov_core {

/**
 * @brief KLT tracking of features.
 *
 * This is the implementation of a KLT visual frontend for tracking sparse features.
 * We can track either monocular cameras across time (temporally) along with
 * stereo cameras which we also track across time (temporally) but track from left to right
 * to find the stereo correspondence information also.
 * This uses the [calcOpticalFlowPyrLK](https://github.com/opencv/opencv/blob/master/modules/video/src/lkpyramid.cpp)
 * OpenCV function to do the KLT tracking.
 */
class TrackKLT : public TrackBase {

public:
  /**
   * @brief Public constructor with configuration variables
   * @param cameras camera calibration object which has all camera intrinsics in it
   * @param numfeats number of features we want want to track (i.e. track 200 points from frame to frame)
   * @param numaruco the max id of the arucotags, so we ensure that we start our non-auroc features above this value
   * @param stereo if we should do stereo feature tracking or binocular
   * @param histmethod what type of histogram pre-processing should be done (histogram eq?)
   * @param fast_threshold FAST detection threshold
   * @param gridx size of grid in the x-direction / u-direction
   * @param gridy size of grid in the y-direction / v-direction
   * @param minpxdist features need to be at least this number pixels away from each other
   * @param pyramid_levels maximum optical-flow pyramid level (zero tracks at the original resolution)
   * @param window_size odd optical-flow patch width in pixels
   * @param max_fb_error maximum forward/backward optical-flow error in pixels; zero disables the check
   * @param ransac_valid_only exclude failed optical-flow predictions before fitting the geometric model
   */
  explicit TrackKLT(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, bool stereo,
                    HistogramMethod histmethod, int fast_threshold, int gridx, int gridy, int minpxdist, int pyramid_levels = 5,
                    int window_size = 15, double max_fb_error = 0.0, bool ransac_valid_only = false)
      : TrackBase(cameras, numfeats, numaruco, stereo, histmethod), threshold(fast_threshold), grid_x(gridx), grid_y(gridy),
        min_px_dist(minpxdist), pyr_levels(pyramid_levels), win_size(window_size, window_size), max_fb_error(max_fb_error),
        ransac_valid_only(ransac_valid_only) {
    if (pyramid_levels < 0 || pyramid_levels > 8 || window_size < 3 || window_size > 63 || window_size % 2 == 0)
      throw std::invalid_argument("KLT pyramid level must be 0..8 and window width must be odd, 3..63 pixels");
    if (!std::isfinite(max_fb_error) || max_fb_error < 0.0)
      throw std::invalid_argument("KLT forward/backward error must be finite and nonnegative");
    for (const auto &camera : camera_calib) {
      img_pyramid_last[camera.first];
      img_curr[camera.first];
      img_pyramid_curr[camera.first];
      if (histmethod == HistogramMethod::CLAHE)
        clahe[camera.first] = cv::createCLAHE(10.0, cv::Size(8, 8));
    }
  }

  /**
   * @brief Process a new image
   * @param message Contains our timestamp, images, and camera ids
   */
  void feed_new_camera(const CameraData &message) override;

  /// Startup selection; calibration may be refreshed between camera feeds by the estimator.
  void set_observation_uncertainty(bool enabled, const std::map<size_t, Eigen::VectorXd> &extrinsics);

  /// Select a complete trained sparse-tracking graph. Failure to load is fatal; there is no silent LK fallback.
  void set_neural_tracker(const std::string &model_path, const std::string &target, double min_confidence);

  /// Use LET-Net corner scores and learned image features for detection and temporal/stereo LK.
  void set_letnet_tracker(const std::string &model_path, double minimum_score, const std::string &engine = "opencv", int threads = 4);

protected:
  /// One camera's tracking result. New features carry pending ids until commit_monocular() applies the result.
  struct MonocularTrack {
    size_t cam_id = 0;
    cv::Mat img, mask;
    std::vector<cv::Mat> imgpyr;
    std::vector<cv::KeyPoint> pts;
    std::vector<size_t> ids;
    size_t detected = 0;          ///< pending ids handed out by detection
    bool update_database = false; ///< false when tracking (re)started in this image
    std::vector<cv::Point2f> source_points;
    std::vector<FlowQuality> quality;
  };

  /**
   * @brief Process a new monocular image
   * @param message Contains our timestamp, images, and camera ids
   * @param msg_id the camera index in message data vector
   */
  void feed_monocular(const CameraData &message, size_t msg_id);

  /// Track one camera without touching shared state, so cameras can be tracked in parallel
  MonocularTrack track_monocular(const CameraData &message, size_t msg_id);

  /// Assign feature ids, update the database, and store the tracks; call in camera order
  void commit_monocular(const CameraData &message, MonocularTrack &track);

  /**
   * @brief Process new stereo pair of images
   * @param message Contains our timestamp, images, and camera ids
   * @param msg_id_left first image index in message data vector
   * @param msg_id_right second image index in message data vector
   */
  void feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right);

  /**
   * @brief Detects new features in the current image
   * @param img0pyr image we will detect features on (first level of pyramid)
   * @param mask0 mask which has what ROI we do not want features in
   * @param pts0 vector of currently extracted keypoints in this image
   * @param ids0 vector of feature ids for each currently extracted keypoint
   * @param detected number of pending ids handed out to new features (incremented)
   *
   * Given an image and its currently extracted features, this will try to add new features if needed.
   * Will try to always have the "max_features" being tracked through KLT at each timestep.
   * Passed images should already be grayscaled.
   */
  void perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                   std::vector<size_t> &ids0, size_t &detected, const cv::Mat *score = nullptr);

  /**
   * @brief Detects new features in the current stereo pair
   * @param img0pyr left image we will detect features on (first level of pyramid)
   * @param img1pyr right image we will detect features on (first level of pyramid)
   * @param mask0 mask which has what ROI we do not want features in
   * @param mask1 mask which has what ROI we do not want features in
   * @param cam_id_left first camera sensor id
   * @param cam_id_right second camera sensor id
   * @param pts0 left vector of currently extracted keypoints
   * @param pts1 right vector of currently extracted keypoints
   * @param ids0 left vector of feature ids for each currently extracted keypoint
   * @param ids1 right vector of feature ids for each currently extracted keypoint
   *
   * This does the same logic as the perform_detection_monocular() function, but we also enforce stereo contraints.
   * So we detect features in the left image, and then KLT track them onto the right image.
   * If we have valid tracks, then we have both the keypoint on the left and its matching point in the right image.
   * Will try to always have the "max_features" being tracked through KLT at each timestep.
   */
  void perform_detection_stereo(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, const cv::Mat &mask0,
                                const cv::Mat &mask1, size_t cam_id_left, size_t cam_id_right, std::vector<cv::KeyPoint> &pts0,
                                std::vector<cv::KeyPoint> &pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1,
                                const cv::Mat *score0 = nullptr, const cv::Mat *score1 = nullptr);

  /**
   * @brief KLT track between two images, and do RANSAC afterwards
   * @param img0pyr starting image pyramid
   * @param img1pyr image pyramid we want to track too
   * @param pts0 starting points
   * @param pts1 points we have tracked
   * @param id0 id of the first camera
   * @param id1 id of the second camera
   * @param mask_out what points had valid tracks
   *
   * This will track features from the first image into the second image.
   * The two point vectors will be of equal size, but the mask_out variable will specify which points are good or bad.
   * If the second vector is non-empty, it will be used as an initial guess of where the keypoints are in the second image.
   */
  void perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &pts0,
                        std::vector<cv::KeyPoint> &pts1, size_t id0, size_t id1, std::vector<uchar> &mask_out,
                        std::vector<FlowQuality> *diagnostics = nullptr, const std::vector<size_t> *feature_ids = nullptr);

  std::vector<Eigen::Matrix2d> observation_covariance(size_t camera, const cv::Mat &previous, const cv::Mat &current,
                                                    const std::vector<cv::Point2f> &old_points,
                                                    const std::vector<cv::KeyPoint> &points, const std::vector<size_t> &ids,
                                                    const std::vector<FlowQuality> &diagnostics);

  /// Fit the epipolar model using only successful, finite, in-image optical-flow correspondences.
  /// The returned mask retains the original indexing, including rejected entries.
  std::vector<uchar> geometric_inlier_mask(const std::vector<cv::Point2f> &pts0, const std::vector<cv::Point2f> &pts1,
                                           const std::vector<uchar> &flow_status, size_t id0, size_t id1, bool valid_only = true);

  // Parameters for our FAST grid detector
  int threshold;
  int grid_x;
  int grid_y;

  // Minimum pixel distance to be "far away enough" to be a different extracted feature
  int min_px_dist;

  // How many pyramid levels to track
  int pyr_levels = 5;
  cv::Size win_size = cv::Size(15, 15);

  /// Round-trip consistency threshold in processed-image pixels; zero preserves one-way tracking.
  double max_fb_error = 0.0;

  /// Experimental geometric-model input filtering. Leave off until trajectory-level validation supports enabling it.
  bool ransac_valid_only = false;

  bool observation_uncertainty = false;
  std::shared_ptr<NeuralFlow> neural_flow;
  double neural_min_confidence = 0.1;
  std::shared_ptr<LearnedFeatures> learned_features;
  double letnet_minimum_score = 0.2;
  std::map<size_t, cv::Mat> score_current, score_last;
  void advance_buffers(size_t camera);
  std::map<size_t, Eigen::Matrix4d> quality_extrinsics;
  std::map<size_t, std::unordered_map<size_t, size_t>> quality_ages;

  // Two independent image/pyramid buffers per camera, swapped after tracking.
  // Never overwrite the previous frame while optical flow still references it.
  std::map<size_t, std::vector<cv::Mat>> img_pyramid_last;
  std::map<size_t, cv::Mat> img_curr;
  std::map<size_t, std::vector<cv::Mat>> img_pyramid_curr;
  std::map<size_t, cv::Ptr<cv::CLAHE>> clahe;
};

} // namespace ov_core

#endif /* OV_CORE_TRACK_KLT_H */
