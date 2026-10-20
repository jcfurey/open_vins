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

#include "TrackKLT.h"

#include <unordered_set>

#include "Grider_FAST.h"
#include "Grider_GRID.h"
#include "cam/CamBase.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "utils/opencv_lambda_body.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

using namespace ov_core;

void TrackKLT::set_letnet_tracker(const std::string &model_path, double minimum_score) {
  if (histogram_method != HistogramMethod::NONE || neural_flow || observation_uncertainty)
    throw std::invalid_argument("LET-Net requires histogram_method=NONE and a separate native covariance/KLTNet selection");
  if (!std::isfinite(minimum_score) || minimum_score < 0 || minimum_score > 1)
    throw std::invalid_argument("letnet_min_score must be in [0,1]");
  learned_features = std::make_shared<LearnedFeatures>(model_path);
  letnet_minimum_score = minimum_score;
  for (const auto &camera : camera_calib) {
    score_current[camera.first];
    score_last[camera.first];
  }
}

void TrackKLT::advance_buffers(size_t camera) {
  std::swap(img_last.at(camera), img_curr.at(camera));
  img_pyramid_last.at(camera).swap(img_pyramid_curr.at(camera));
  if (learned_features)
    std::swap(score_last.at(camera), score_current.at(camera));
}

void TrackKLT::set_neural_tracker(const std::string &model_path, const std::string &target, double min_confidence) {
  if (learned_features)
    throw std::invalid_argument("Select only one learned tracking frontend");
  if (!std::isfinite(min_confidence) || min_confidence < 0)
    throw std::invalid_argument("kltnet_min_confidence must be finite and nonnegative");
  neural_flow = std::make_shared<NeuralFlow>(model_path, target);
  neural_min_confidence = min_confidence;
}

void TrackKLT::set_observation_uncertainty(bool enabled, const std::map<size_t, Eigen::VectorXd> &extrinsics) {
  if (enabled && learned_features)
    throw std::invalid_argument("RDIC patch uncertainty currently expects grayscale native LK features");
  observation_uncertainty = enabled;
  if (!enabled)
    return;
  for (const auto &camera : camera_calib) {
    const auto &pose = extrinsics.at(camera.first);
    if (pose.size() != 7 || !pose.allFinite())
      throw std::invalid_argument("Observation quality requires valid camera extrinsics");
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.topLeftCorner<3, 3>() = quat_2_Rot(pose.head<4>());
    transform.topRightCorner<3, 1>() = pose.tail<3>();
    quality_extrinsics[camera.first] = transform;
    quality_ages[camera.first];
  }
}

std::vector<Eigen::Matrix2d> TrackKLT::observation_covariance(size_t camera, const cv::Mat &previous, const cv::Mat &current,
                                                           const std::vector<cv::Point2f> &old_points,
                                                           const std::vector<cv::KeyPoint> &points, const std::vector<size_t> &ids,
                                                           const std::vector<FlowQuality> &diagnostics) {
  std::vector<cv::Point2f> new_points;
  std::vector<size_t> ages;
  std::unordered_map<size_t, size_t> next_ages;
  const auto &previous_ages = quality_ages.at(camera);
  for (size_t i = 0; i < ids.size(); ++i) {
    auto old = previous_ages.find(ids[i]);
    const size_t age = old == previous_ages.end() ? 1 : old->second + 1;
    ages.push_back(age);
    next_ages.emplace(ids[i], age);
    new_points.push_back(points[i].pt);
  }
  auto result = TrackQuality::covariance(previous, current, old_points, new_points, diagnostics, ages);
  quality_ages.at(camera).swap(next_ages);
  return result;
}

void TrackKLT::feed_new_camera(const CameraData &message) {

  // Error check that we have all the data
  if (message.sensor_ids.empty() || message.sensor_ids.size() != message.images.size() || message.images.size() != message.masks.size()) {
    PRINT_ERROR(RED "[ERROR]: MESSAGE DATA SIZES DO NOT MATCH OR EMPTY!!!\n" RESET);
    PRINT_ERROR(RED "[ERROR]:   - message.sensor_ids.size() = %zu\n" RESET, message.sensor_ids.size());
    PRINT_ERROR(RED "[ERROR]:   - message.images.size() = %zu\n" RESET, message.images.size());
    PRINT_ERROR(RED "[ERROR]:   - message.masks.size() = %zu\n" RESET, message.masks.size());
    std::exit(EXIT_FAILURE);
  }

  // Preprocessing steps that we do not parallelize
  // NOTE: DO NOT PARALLELIZE THESE!
  // NOTE: These seem to be much slower if you parallelize them...
  rT1 = boost::posix_time::microsec_clock::local_time();
  size_t num_images = message.images.size();
  for (size_t msg_id = 0; msg_id < num_images; msg_id++) {

    // Lock this data feed for this camera
    size_t cam_id = message.sensor_ids.at(msg_id);
    std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

    // Histogram equalize
    cv::Mat &img = img_curr.at(cam_id);
    if (histogram_method == HistogramMethod::HISTOGRAM) {
      cv::equalizeHist(message.images.at(msg_id), img);
    } else if (histogram_method == HistogramMethod::CLAHE) {
      clahe.at(cam_id)->apply(message.images.at(msg_id), img);
    } else {
      img = message.images.at(msg_id);
    }

    // Extract image pyramid
    if (learned_features) {
      auto features = learned_features->infer(img);
      score_current.at(cam_id) = std::move(features.score);
      cv::buildOpticalFlowPyramid(features.tracking, img_pyramid_curr.at(cam_id), win_size, pyr_levels);
    } else {
      cv::buildOpticalFlowPyramid(img, img_pyramid_curr.at(cam_id), win_size, pyr_levels);
    }
  }

  // Either call our stereo or monocular version
  // If we are doing binocular tracking, then we should parallize our tracking
  if (num_images == 1) {
    feed_monocular(message, 0);
  } else if (num_images == 2 && use_stereo) {
    feed_stereo(message, 0, 1);
  } else if (!use_stereo) {
    // Track cameras in parallel, then commit them in camera order so feature ids do not depend on thread scheduling
    std::vector<MonocularTrack> tracks(num_images);
    parallel_for_(cv::Range(0, (int)num_images), LambdaBody([&](const cv::Range &range) {
                    for (int i = range.start; i < range.end; i++) {
                      tracks.at(i) = track_monocular(message, i);
                    }
                  }));
    for (auto &track : tracks)
      commit_monocular(message, track);
  } else {
    PRINT_ERROR(RED "[ERROR]: invalid number of images passed %zu, we only support mono or stereo tracking", num_images);
    std::exit(EXIT_FAILURE);
  }
}

void TrackKLT::feed_monocular(const CameraData &message, size_t msg_id) {
  MonocularTrack track = track_monocular(message, msg_id);
  commit_monocular(message, track);
}

TrackKLT::MonocularTrack TrackKLT::track_monocular(const CameraData &message, size_t msg_id) {

  // Lock this data feed for this camera
  MonocularTrack track;
  track.cam_id = message.sensor_ids.at(msg_id);
  const size_t cam_id = track.cam_id;
  std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

  // Get our image objects for this image
  track.img = img_curr.at(cam_id);
  track.imgpyr = img_pyramid_curr.at(cam_id);
  track.mask = message.masks.at(msg_id);
  const auto time_start = boost::posix_time::microsec_clock::local_time();

  // If we didn't have any successful tracks last time, just extract this time
  // This also handles, the tracking initalization on the first call to this extractor
  if (pts_last.at(cam_id).empty()) {
    perform_detection_monocular(track.imgpyr, track.mask, track.pts, track.ids, track.detected,
                                 learned_features ? &score_current.at(cam_id) : nullptr);
    return track;
  }

  // First we should make that the last images have enough features so we can do KLT
  // This will "top-off" our number of tracks so always have a constant number
  auto pts_left_old = pts_last.at(cam_id);
  auto ids_left_old = ids_last.at(cam_id);
  perform_detection_monocular(img_pyramid_last.at(cam_id), img_mask_last.at(cam_id), pts_left_old, ids_left_old, track.detected,
                               learned_features ? &score_last.at(cam_id) : nullptr);
  const auto time_detect = boost::posix_time::microsec_clock::local_time();

  // Our return success masks, and predicted new features
  std::vector<uchar> mask_ll;
  std::vector<cv::KeyPoint> pts_left_new = pts_left_old;
  std::vector<FlowQuality> diagnostics;

  // Lets track temporally
  perform_matching(img_pyramid_last.at(cam_id), track.imgpyr, pts_left_old, pts_left_new, cam_id, cam_id, mask_ll,
                   observation_uncertainty ? &diagnostics : nullptr, &ids_left_old);
  assert(pts_left_new.size() == ids_left_old.size());
  const auto time_klt = boost::posix_time::microsec_clock::local_time();

  // If any of our mask is empty, that means we didn't have enough to do ransac, so just return
  if (mask_ll.empty()) {
    PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
    return track;
  }

  // Get our "good tracks"
  for (size_t i = 0; i < pts_left_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x >= track.img.cols ||
        (int)pts_left_new.at(i).pt.y >= track.img.rows)
      continue;
    // Check if it is in the mask
    // NOTE: mask has max value of 255 (white) if it should be
    if ((int)message.masks.at(msg_id).at<uint8_t>((int)pts_left_new.at(i).pt.y, (int)pts_left_new.at(i).pt.x) > 127)
      continue;
    // If it is a good track, and also tracked from left to right
    if (mask_ll[i]) {
      track.pts.push_back(pts_left_new[i]);
      track.ids.push_back(ids_left_old[i]);
      if (observation_uncertainty) {
        track.source_points.push_back(pts_left_old[i].pt);
        track.quality.push_back(diagnostics[i]);
      }
    }
  }
  track.update_database = true;

  // Timing information
  PRINT_ALL("[TIME-KLT]: %.4f seconds for detection (%zu detected)\n", (time_detect - time_start).total_microseconds() * 1e-6,
            track.detected);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for temporal klt\n", (time_klt - time_detect).total_microseconds() * 1e-6);
  return track;
}

void TrackKLT::commit_monocular(const CameraData &message, MonocularTrack &track) {

  // Lock this data feed for this camera
  const size_t cam_id = track.cam_id;
  std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));
  const auto previous_ids = neural_flow ? track.ids : std::vector<size_t>();
  resolve_pending_ids(track.ids, track.detected);
  if (neural_flow) {
    for (size_t i = 0; i < track.ids.size(); ++i)
      neural_flow->rekey(cam_id, previous_ids[i], track.ids[i]);
    neural_flow->retain(cam_id, track.ids);
  }

  // Update our feature database, with theses new observations
  if (track.update_database) {
    const std::vector<cv::Point2f> pts_n = undistort_keypoints(cam_id, track.pts);
    std::vector<Eigen::Matrix2d> covariances;
    if (observation_uncertainty)
      covariances = observation_covariance(cam_id, img_pyramid_last.at(cam_id)[0], track.img, track.source_points,
                                           track.pts, track.ids, track.quality);
    for (size_t i = 0; i < track.pts.size(); i++) {
      database->update_feature(track.ids.at(i), message.timestamp, cam_id, track.pts.at(i).pt.x, track.pts.at(i).pt.y, pts_n.at(i).x,
                               pts_n.at(i).y, observation_uncertainty ? &covariances[i] : nullptr);
    }
  }

  // Move forward in time
  std::lock_guard<std::mutex> lckv(mtx_last_vars);
  advance_buffers(cam_id);
  img_mask_last.at(cam_id) = track.mask;
  pts_last.at(cam_id) = track.pts;
  ids_last.at(cam_id) = track.ids;
}

void TrackKLT::feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right) {

  // Lock this data feed for this camera
  size_t cam_id_left = message.sensor_ids.at(msg_id_left);
  size_t cam_id_right = message.sensor_ids.at(msg_id_right);
  std::lock_guard<std::mutex> lck1(mtx_feeds.at(cam_id_left));
  std::lock_guard<std::mutex> lck2(mtx_feeds.at(cam_id_right));

  // Get our image objects for this image
  cv::Mat img_left = img_curr.at(cam_id_left);
  cv::Mat img_right = img_curr.at(cam_id_right);
  std::vector<cv::Mat> imgpyr_left = img_pyramid_curr.at(cam_id_left);
  std::vector<cv::Mat> imgpyr_right = img_pyramid_curr.at(cam_id_right);
  cv::Mat mask_left = message.masks.at(msg_id_left);
  cv::Mat mask_right = message.masks.at(msg_id_right);
  rT2 = boost::posix_time::microsec_clock::local_time();

  // If we didn't have any successful tracks last time, just extract this time
  // This also handles, the tracking initalization on the first call to this extractor
  if (pts_last[cam_id_left].empty() && pts_last[cam_id_right].empty()) {
    // Track into the new image
    std::vector<cv::KeyPoint> good_left, good_right;
    std::vector<size_t> good_ids_left, good_ids_right;
    perform_detection_stereo(imgpyr_left, imgpyr_right, mask_left, mask_right, cam_id_left, cam_id_right, good_left, good_right,
                             good_ids_left, good_ids_right,
                             learned_features ? &score_current.at(cam_id_left) : nullptr,
                             learned_features ? &score_current.at(cam_id_right) : nullptr);
    // Save the current image and pyramid
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    advance_buffers(cam_id_left);
    advance_buffers(cam_id_right);
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left] = good_left;
    pts_last[cam_id_right] = good_right;
    ids_last[cam_id_left] = good_ids_left;
    ids_last[cam_id_right] = good_ids_right;
    return;
  }

  // First we should make that the last images have enough features so we can do KLT
  // This will "top-off" our number of tracks so always have a constant number
  int pts_before_detect = (int)pts_last[cam_id_left].size();
  auto pts_left_old = pts_last[cam_id_left];
  auto pts_right_old = pts_last[cam_id_right];
  auto ids_left_old = ids_last[cam_id_left];
  auto ids_right_old = ids_last[cam_id_right];
  perform_detection_stereo(img_pyramid_last[cam_id_left], img_pyramid_last[cam_id_right], img_mask_last[cam_id_left],
                           img_mask_last[cam_id_right], cam_id_left, cam_id_right, pts_left_old, pts_right_old, ids_left_old,
                           ids_right_old, learned_features ? &score_last.at(cam_id_left) : nullptr,
                           learned_features ? &score_last.at(cam_id_right) : nullptr);
  rT3 = boost::posix_time::microsec_clock::local_time();

  // Our return success masks, and predicted new features
  std::vector<uchar> mask_ll, mask_rr;
  std::vector<FlowQuality> quality_left, quality_right;
  std::vector<cv::KeyPoint> pts_left_new = pts_left_old;
  std::vector<cv::KeyPoint> pts_right_new = pts_right_old;

  // Lets track temporally
  parallel_for_(cv::Range(0, 2), LambdaBody([&](const cv::Range &range) {
                  for (int i = range.start; i < range.end; i++) {
                    bool is_left = (i == 0);
                    perform_matching(img_pyramid_last[is_left ? cam_id_left : cam_id_right], is_left ? imgpyr_left : imgpyr_right,
                                     is_left ? pts_left_old : pts_right_old, is_left ? pts_left_new : pts_right_new,
                                     is_left ? cam_id_left : cam_id_right, is_left ? cam_id_left : cam_id_right,
                                     is_left ? mask_ll : mask_rr,
                                     observation_uncertainty ? (is_left ? &quality_left : &quality_right) : nullptr,
                                     is_left ? &ids_left_old : &ids_right_old);
                  }
                }));
  rT4 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  //===================================================================================

  // left to right matching
  // TODO: we should probably still do this to reject outliers
  // TODO: maybe we should collect all tracks that are in both frames and make they pass this?
  // std::vector<uchar> mask_lr;
  // perform_matching(imgpyr_left, imgpyr_right, pts_left_new, pts_right_new, cam_id_left, cam_id_right, mask_lr);
  rT5 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  //===================================================================================

  // If any of our masks are empty, that means we didn't have enough to do ransac, so just return
  if (mask_ll.empty() && mask_rr.empty()) {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    advance_buffers(cam_id_left);
    advance_buffers(cam_id_right);
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left].clear();
    pts_last[cam_id_right].clear();
    ids_last[cam_id_left].clear();
    ids_last[cam_id_right].clear();
    if (neural_flow) {
      neural_flow->retain(cam_id_left, {});
      neural_flow->retain(cam_id_right, {});
    }
    PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
    return;
  }

  // Get our "good tracks"
  std::vector<cv::KeyPoint> good_left, good_right;
  std::vector<size_t> good_ids_left, good_ids_right;

  // Keep original track order, with constant-time ID lookup in the other camera.
  std::unordered_map<size_t, size_t> right_indices;
  right_indices.reserve(ids_right_old.size());
  for (size_t i = 0; i < ids_right_old.size(); ++i)
    right_indices.emplace(ids_right_old[i], i);
  std::unordered_set<size_t> added_right_ids;
  added_right_ids.reserve(ids_right_old.size());

  // Loop through all left points
  for (size_t i = 0; i < pts_left_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x >= img_left.cols ||
        (int)pts_left_new.at(i).pt.y >= img_left.rows)
      continue;
    // See if we have the same feature in the right
    const auto right = right_indices.find(ids_left_old.at(i));
    const bool found_right = right != right_indices.end();
    const size_t index_right = found_right ? right->second : 0;
    // If it is a good track, and also tracked from left to right
    // Else track it as a mono feature in just the left image
    if (mask_ll[i] && found_right && mask_rr[index_right]) {
      // Ensure we do not have any bad KLT tracks (i.e., points are negative)
      if (pts_right_new.at(index_right).pt.x < 0 || pts_right_new.at(index_right).pt.y < 0 ||
          (int)pts_right_new.at(index_right).pt.x >= img_right.cols || (int)pts_right_new.at(index_right).pt.y >= img_right.rows)
        continue;
      good_left.push_back(pts_left_new.at(i));
      good_right.push_back(pts_right_new.at(index_right));
      good_ids_left.push_back(ids_left_old.at(i));
      good_ids_right.push_back(ids_right_old.at(index_right));
      added_right_ids.insert(ids_right_old.at(index_right));
      // PRINT_DEBUG("adding to stereo - %u , %u\n", ids_left_old.at(i), ids_right_old.at(index_right));
    } else if (mask_ll[i]) {
      good_left.push_back(pts_left_new.at(i));
      good_ids_left.push_back(ids_left_old.at(i));
      // PRINT_DEBUG("adding to left - %u \n",ids_left_old.at(i));
    }
  }

  // Loop through all right points
  for (size_t i = 0; i < pts_right_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_right_new.at(i).pt.x < 0 || pts_right_new.at(i).pt.y < 0 || (int)pts_right_new.at(i).pt.x >= img_right.cols ||
        (int)pts_right_new.at(i).pt.y >= img_right.rows)
      continue;
    // See if we have the same feature in the right
    bool added_already = added_right_ids.find(ids_right_old.at(i)) != added_right_ids.end();
    // If it has not already been added as a good feature, add it as a mono track
    if (mask_rr[i] && !added_already) {
      good_right.push_back(pts_right_new.at(i));
      good_ids_right.push_back(ids_right_old.at(i));
      added_right_ids.insert(ids_right_old.at(i));
      // PRINT_DEBUG("adding to right - %u \n", ids_right_old.at(i));
    }
  }

  std::vector<Eigen::Matrix2d> covariance_left, covariance_right;
  if (observation_uncertainty) {
    std::unordered_map<size_t, size_t> left_indices, good_right_indices;
    for (size_t i = 0; i < ids_left_old.size(); ++i)
      left_indices.emplace(ids_left_old[i], i);
    std::vector<cv::Point2f> source_left, source_right;
    std::vector<FlowQuality> diagnostics_left, diagnostics_right;
    for (size_t i = 0; i < good_ids_left.size(); ++i) {
      const size_t index = left_indices.at(good_ids_left[i]);
      source_left.push_back(pts_left_old[index].pt);
      diagnostics_left.push_back(quality_left[index]);
    }
    for (size_t i = 0; i < good_ids_right.size(); ++i) {
      const size_t index = right_indices.at(good_ids_right[i]);
      source_right.push_back(pts_right_old[index].pt);
      diagnostics_right.push_back(quality_right[index]);
      good_right_indices.emplace(good_ids_right[i], i);
    }
    const Eigen::Matrix4d relative = quality_extrinsics.at(cam_id_right) * quality_extrinsics.at(cam_id_left).inverse();
    const Eigen::Matrix3d rotation = relative.topLeftCorner<3, 3>();
    const Eigen::Matrix3d essential = skew_x(relative.topRightCorner<3, 1>()) * rotation;
    const double focal = std::max(camera_calib.at(cam_id_left)->get_K()(0, 0), camera_calib.at(cam_id_right)->get_K()(0, 0));
    auto bearing = [&](size_t camera, const cv::Point2f &point) {
      const auto p = camera_calib.at(camera)->undistort_cv(point);
      return Eigen::Vector3d(p.x, p.y, 1.0);
    };
    for (size_t left = 0; left < good_ids_left.size(); ++left) {
      auto found = good_right_indices.find(good_ids_left[left]);
      if (found == good_right_indices.end())
        continue;
      const size_t right = found->second;
      const auto l = bearing(cam_id_left, good_left[left].pt), r = bearing(cam_id_right, good_right[right].pt);
      const auto old_l = bearing(cam_id_left, source_left[left]), old_r = bearing(cam_id_right, source_right[right]);
      const double epipolar = TrackQuality::epipolar_error(essential, l, r, focal);
      const double imbalance = std::abs(diagnostics_left[left].temporal - diagnostics_right[right].temporal);
      const Eigen::Vector3d rotated_l = rotation * l, rotated_old_l = rotation * old_l;
      const double disparity = focal * std::abs((r.head<2>() - rotated_l.head<2>() / rotated_l.z()).norm() -
                                                (old_r.head<2>() - rotated_old_l.head<2>() / rotated_old_l.z()).norm());
      diagnostics_left[left].balance = diagnostics_right[right].balance = imbalance;
      diagnostics_left[left].stereo = diagnostics_right[right].stereo = epipolar;
      diagnostics_left[left].disparity = diagnostics_right[right].disparity = disparity;
    }
    covariance_left = observation_covariance(cam_id_left, img_pyramid_last.at(cam_id_left)[0], img_left,
                                             source_left, good_left, good_ids_left, diagnostics_left);
    covariance_right = observation_covariance(cam_id_right, img_pyramid_last.at(cam_id_right)[0], img_right,
                                              source_right, good_right, good_ids_right, diagnostics_right);
  }

  if (neural_flow) {
    neural_flow->retain(cam_id_left, good_ids_left);
    neural_flow->retain(cam_id_right, good_ids_right);
  }

  // Update our feature database, with theses new observations
  const std::vector<cv::Point2f> good_left_n = undistort_keypoints(cam_id_left, good_left);
  for (size_t i = 0; i < good_left.size(); i++) {
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id_left, good_left.at(i).pt.x, good_left.at(i).pt.y,
                             good_left_n.at(i).x, good_left_n.at(i).y, observation_uncertainty ? &covariance_left[i] : nullptr);
  }
  const std::vector<cv::Point2f> good_right_n = undistort_keypoints(cam_id_right, good_right);
  for (size_t i = 0; i < good_right.size(); i++) {
    database->update_feature(good_ids_right.at(i), message.timestamp, cam_id_right, good_right.at(i).pt.x, good_right.at(i).pt.y,
                             good_right_n.at(i).x, good_right_n.at(i).y, observation_uncertainty ? &covariance_right[i] : nullptr);
  }

  // Move forward in time
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    advance_buffers(cam_id_left);
    advance_buffers(cam_id_right);
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left] = good_left;
    pts_last[cam_id_right] = good_right;
    ids_last[cam_id_left] = good_ids_left;
    ids_last[cam_id_right] = good_ids_right;
  }
  rT6 = boost::posix_time::microsec_clock::local_time();

  //  // Timing information
  PRINT_ALL("[TIME-KLT]: %.4f seconds for pyramid\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for detection (%d detected)\n", (rT3 - rT2).total_microseconds() * 1e-6,
            (int)pts_last[cam_id_left].size() - pts_before_detect);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for temporal klt\n", (rT4 - rT3).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for stereo klt\n", (rT5 - rT4).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for feature DB update (%d features)\n", (rT6 - rT5).total_microseconds() * 1e-6,
            (int)good_left.size());
  PRINT_ALL("[TIME-KLT]: %.4f seconds for total\n", (rT6 - rT1).total_microseconds() * 1e-6);
}

void TrackKLT::perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                           std::vector<size_t> &ids0, size_t &detected, const cv::Mat *score) {

  // Create a 2D occupancy grid for this current image
  // Note that we scale this down, so that each grid point is equal to a set of pixels
  // This means that we will reject points that less than grid_px_size points away then existing features
  cv::Size size_close((int)((float)img0pyr.at(0).cols / (float)min_px_dist),
                      (int)((float)img0pyr.at(0).rows / (float)min_px_dist)); // width x height
  cv::Mat grid_2d_close = cv::Mat::zeros(size_close, CV_8UC1);
  float size_x = (float)img0pyr.at(0).cols / (float)grid_x;
  float size_y = (float)img0pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid(grid_x, grid_y); // width x height
  cv::Mat grid_2d_grid = cv::Mat::zeros(size_grid, CV_8UC1);
  cv::Mat mask0_updated = mask0.clone();
  auto it0 = pts0.begin();
  auto it1 = ids0.begin();
  while (it0 != pts0.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img0pyr.at(0).cols - edge || y < edge || y >= img0pyr.at(0).rows - edge) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate mask coordinates for close points
    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close.width || y_close < 0 || y_close >= size_close.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate what grid cell this feature is in
    int x_grid = std::floor(kpt.pt.x / size_x);
    int y_grid = std::floor(kpt.pt.y / size_y);
    if (x_grid < 0 || x_grid >= size_grid.width || y_grid < 0 || y_grid >= size_grid.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Check if this keypoint is near another point
    if (grid_2d_close.at<uint8_t>(y_close, x_close) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask0.at<uint8_t>(y, x) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_close.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid.at<uint8_t>(y_grid, x_grid) += 1;
    }
    // Append this to the local mask of the image
    if (x - min_px_dist >= 0 && x + min_px_dist < img0pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img0pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist, y - min_px_dist);
      cv::Point pt2(x + min_px_dist, y + min_px_dist);
      cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255), -1);
    }
    it0++;
    it1++;
  }

  // First compute how many more features we need to extract from this image
  // If we don't need any features, just return
  double min_feat_percent = 0.50;
  int num_featsneeded = num_features - (int)pts0.size();
  if (num_featsneeded < std::min(20, (int)(min_feat_percent * num_features)))
    return;

  // This is old extraction code that would extract from the whole image
  // This can be slow as this will recompute extractions for grid areas that we have max features already
  // std::vector<cv::KeyPoint> pts0_ext;
  // Grider_FAST::perform_griding(img0pyr.at(0), mask0_updated, pts0_ext, num_features, grid_x, grid_y, threshold, true);

  // We also check a downsampled mask such that we don't extract in areas where it is all masked!
  cv::Mat mask0_grid;
  cv::resize(mask0, mask0_grid, size_grid, 0.0, 0.0, cv::INTER_NEAREST);

  // Create grids we need to extract from and then extract our features (use fast with griding)
  int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
  int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
  std::vector<std::pair<int, int>> valid_locs;
  for (int x = 0; x < grid_2d_grid.cols; x++) {
    for (int y = 0; y < grid_2d_grid.rows; y++) {
      if ((int)grid_2d_grid.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) != 255) {
        valid_locs.emplace_back(x, y);
      }
    }
  }
  std::vector<cv::KeyPoint> pts0_ext;
  if (score)
    LearnedFeatures::detect(*score, mask0_updated, valid_locs, pts0_ext, num_features_grid, grid_x, grid_y, letnet_minimum_score);
  else
    Grider_GRID::perform_griding(img0pyr.at(0), mask0_updated, valid_locs, pts0_ext, num_features, grid_x, grid_y, threshold, true);

  // Now, reject features that are close a current feature
  std::vector<cv::KeyPoint> kpts0_new;
  std::vector<cv::Point2f> pts0_new;
  for (auto &kpt : pts0_ext) {
    // Check that it is in bounds
    int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
    int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_grid < 0 || x_grid >= size_close.width || y_grid < 0 || y_grid >= size_close.height)
      continue;
    // See if there is a point at this location
    if (grid_2d_close.at<uint8_t>(y_grid, x_grid) > 127)
      continue;
    // Else lets add it!
    kpts0_new.push_back(kpt);
    pts0_new.push_back(kpt.pt);
    grid_2d_close.at<uint8_t>(y_grid, x_grid) = 255;
  }

  // Loop through and record only ones that are valid
  // NOTE: ids are pending until the caller commits this camera, so parallel cameras cannot interleave them
  for (size_t i = 0; i < pts0_new.size(); i++) {
    // update the uv coordinates
    kpts0_new.at(i).pt = pts0_new.at(i);
    // append the new uv coordinate
    pts0.push_back(kpts0_new.at(i));
    // append this new point with the next pending id
    ids0.push_back(PENDING_ID | detected++);
  }
}

void TrackKLT::perform_detection_stereo(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, const cv::Mat &mask0,
                                        const cv::Mat &mask1, size_t cam_id_left, size_t cam_id_right, std::vector<cv::KeyPoint> &pts0,
                                        std::vector<cv::KeyPoint> &pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1,
                                        const cv::Mat *score0, const cv::Mat *score1) {

  // Create a 2D occupancy grid for this current image
  // Note that we scale this down, so that each grid point is equal to a set of pixels
  // This means that we will reject points that less then grid_px_size points away then existing features
  cv::Size size_close0((int)((float)img0pyr.at(0).cols / (float)min_px_dist),
                       (int)((float)img0pyr.at(0).rows / (float)min_px_dist)); // width x height
  cv::Mat grid_2d_close0 = cv::Mat::zeros(size_close0, CV_8UC1);
  float size_x0 = (float)img0pyr.at(0).cols / (float)grid_x;
  float size_y0 = (float)img0pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid0(grid_x, grid_y); // width x height
  cv::Mat grid_2d_grid0 = cv::Mat::zeros(size_grid0, CV_8UC1);
  cv::Mat mask0_updated = mask0.clone();
  auto it0 = pts0.begin();
  auto it1 = ids0.begin();
  while (it0 != pts0.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img0pyr.at(0).cols - edge || y < edge || y >= img0pyr.at(0).rows - edge) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate mask coordinates for close points
    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close0.width || y_close < 0 || y_close >= size_close0.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate what grid cell this feature is in
    int x_grid = std::floor(kpt.pt.x / size_x0);
    int y_grid = std::floor(kpt.pt.y / size_y0);
    if (x_grid < 0 || x_grid >= size_grid0.width || y_grid < 0 || y_grid >= size_grid0.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Check if this keypoint is near another point
    if (grid_2d_close0.at<uint8_t>(y_close, x_close) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask0.at<uint8_t>(y, x) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_close0.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid0.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid0.at<uint8_t>(y_grid, x_grid) += 1;
    }
    // Append this to the local mask of the image
    if (x - min_px_dist >= 0 && x + min_px_dist < img0pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img0pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist, y - min_px_dist);
      cv::Point pt2(x + min_px_dist, y + min_px_dist);
      cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255), -1);
    }
    it0++;
    it1++;
  }

  // First compute how many more features we need to extract from this image
  double min_feat_percent = 0.50;
  int num_featsneeded_0 = num_features - (int)pts0.size();

  // LEFT: if we need features we should extract them in the current frame
  // LEFT: we will also try to track them from this frame over to the right frame
  // LEFT: in the case that we have two features that are the same, then we should merge them
  if (num_featsneeded_0 > std::min(20, (int)(min_feat_percent * num_features))) {

    // This is old extraction code that would extract from the whole image
    // This can be slow as this will recompute extractions for grid areas that we have max features already
    // std::vector<cv::KeyPoint> pts0_ext;
    // Grider_FAST::perform_griding(img0pyr.at(0), mask0_updated, pts0_ext, num_features, grid_x, grid_y, threshold, true);

    // We also check a downsampled mask such that we don't extract in areas where it is all masked!
    cv::Mat mask0_grid;
    cv::resize(mask0, mask0_grid, size_grid0, 0.0, 0.0, cv::INTER_NEAREST);

    // Create grids we need to extract from and then extract our features (use fast with griding)
    int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
    int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
    std::vector<std::pair<int, int>> valid_locs;
    for (int x = 0; x < grid_2d_grid0.cols; x++) {
      for (int y = 0; y < grid_2d_grid0.rows; y++) {
        if ((int)grid_2d_grid0.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) != 255) {
          valid_locs.emplace_back(x, y);
        }
      }
    }
    std::vector<cv::KeyPoint> pts0_ext;
    if (score0)
      LearnedFeatures::detect(*score0, mask0_updated, valid_locs, pts0_ext, num_features_grid, grid_x, grid_y, letnet_minimum_score);
    else
      Grider_GRID::perform_griding(img0pyr.at(0), mask0_updated, valid_locs, pts0_ext, num_features, grid_x, grid_y, threshold, true);

    // Now, reject features that are close a current feature
    std::vector<cv::KeyPoint> kpts0_new;
    std::vector<cv::Point2f> pts0_new;
    for (auto &kpt : pts0_ext) {
      // Check that it is in bounds
      int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
      int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
      if (x_grid < 0 || x_grid >= size_close0.width || y_grid < 0 || y_grid >= size_close0.height)
        continue;
      // See if there is a point at this location
      if (grid_2d_close0.at<uint8_t>(y_grid, x_grid) > 127)
        continue;
      // Else lets add it!
      grid_2d_close0.at<uint8_t>(y_grid, x_grid) = 255;
      kpts0_new.push_back(kpt);
      pts0_new.push_back(kpt.pt);
    }

    // TODO: Project points from the left frame into the right frame
    // TODO: This will not work for large baseline systems.....
    // TODO: If we had some depth estimates we could do a better projection
    // TODO: Or project and search along the epipolar line??
    std::vector<cv::KeyPoint> kpts1_new;
    std::vector<cv::Point2f> pts1_new;
    kpts1_new = kpts0_new;
    pts1_new = pts0_new;

    // If we have points, do KLT tracking to get the valid projections into the right image
    if (!pts0_new.empty()) {

      // Do our KLT tracking from the left to the right frame of reference
      // NOTE: we have a pretty big window size here since our projection might be bad
      // NOTE: but this might cause failure in cases of repeated textures (eg. checkerboard)
      std::vector<uchar> mask;
      // perform_matching(img0pyr, img1pyr, kpts0_new, kpts1_new, cam_id_left, cam_id_right, mask);
      std::vector<float> error;
      cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
      cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0_new, pts1_new, mask, error, win_size, pyr_levels, term_crit,
                               cv::OPTFLOW_USE_INITIAL_FLOW);

      // Loop through and record only ones that are valid
      for (size_t i = 0; i < pts0_new.size(); i++) {

        // Check to see if the feature is out of bounds (oob) in either image
        bool oob_left = ((int)pts0_new.at(i).x < 0 || (int)pts0_new.at(i).x >= img0pyr.at(0).cols || (int)pts0_new.at(i).y < 0 ||
                         (int)pts0_new.at(i).y >= img0pyr.at(0).rows);
        bool oob_right = ((int)pts1_new.at(i).x < 0 || (int)pts1_new.at(i).x >= img1pyr.at(0).cols || (int)pts1_new.at(i).y < 0 ||
                          (int)pts1_new.at(i).y >= img1pyr.at(0).rows);

        // Check to see if it there is already a feature in the right image at this location
        //  1) If this is not already in the right image, then we should treat it as a stereo
        //  2) Otherwise we will treat this as just a monocular track of the feature
        // TODO: we should check to see if we can combine this new feature and the one in the right
        // TODO: seems if reject features which overlay with right features already we have very poor tracking perf
        if (!oob_left && !oob_right && mask[i] == 1) {
          // update the uv coordinates
          kpts0_new.at(i).pt = pts0_new.at(i);
          kpts1_new.at(i).pt = pts1_new.at(i);
          // append the new uv coordinate
          pts0.push_back(kpts0_new.at(i));
          pts1.push_back(kpts1_new.at(i));
          // move id forward and append this new point
          size_t temp = ++currid;
          ids0.push_back(temp);
          ids1.push_back(temp);
        } else if (!oob_left) {
          // update the uv coordinates
          kpts0_new.at(i).pt = pts0_new.at(i);
          // append the new uv coordinate
          pts0.push_back(kpts0_new.at(i));
          // move id forward and append this new point
          size_t temp = ++currid;
          ids0.push_back(temp);
        }
      }
    }
  }

  // RIGHT: Now summarise the number of tracks in the right image
  // RIGHT: We will try to extract some monocular features if we have the room
  // RIGHT: This will also remove features if there are multiple in the same location
  cv::Size size_close1((int)((float)img1pyr.at(0).cols / (float)min_px_dist), (int)((float)img1pyr.at(0).rows / (float)min_px_dist));
  cv::Mat grid_2d_close1 = cv::Mat::zeros(size_close1, CV_8UC1);
  float size_x1 = (float)img1pyr.at(0).cols / (float)grid_x;
  float size_y1 = (float)img1pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid1(grid_x, grid_y); // width x height
  cv::Mat grid_2d_grid1 = cv::Mat::zeros(size_grid1, CV_8UC1);
  cv::Mat mask1_updated = mask1.clone();
  it0 = pts1.begin();
  it1 = ids1.begin();
  while (it0 != pts1.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img1pyr.at(0).cols - edge || y < edge || y >= img1pyr.at(0).rows - edge) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Calculate mask coordinates for close points
    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close1.width || y_close < 0 || y_close >= size_close1.height) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Calculate what grid cell this feature is in
    int x_grid = std::floor(kpt.pt.x / size_x1);
    int y_grid = std::floor(kpt.pt.y / size_y1);
    if (x_grid < 0 || x_grid >= size_grid1.width || y_grid < 0 || y_grid >= size_grid1.height) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Check if this keypoint is near another point
    // NOTE: if it is *not* a stereo point, then we will not delete the feature
    // NOTE: this means we might have a mono and stereo feature near each other, but that is ok
    bool is_stereo = (std::find(ids0.begin(), ids0.end(), *it1) != ids0.end());
    if (grid_2d_close1.at<uint8_t>(y_close, x_close) > 127 && !is_stereo) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask1.at<uint8_t>(y, x) > 127) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_close1.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid1.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid1.at<uint8_t>(y_grid, x_grid) += 1;
    }
    // Append this to the local mask of the image
    if (x - min_px_dist >= 0 && x + min_px_dist < img1pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img1pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist, y - min_px_dist);
      cv::Point pt2(x + min_px_dist, y + min_px_dist);
      cv::rectangle(mask1_updated, pt1, pt2, cv::Scalar(255), -1);
    }
    it0++;
    it1++;
  }

  // RIGHT: if we need features we should extract them in the current frame
  // RIGHT: note that we don't track them to the left as we already did left->right tracking above
  int num_featsneeded_1 = num_features - (int)pts1.size();
  if (num_featsneeded_1 > std::min(20, (int)(min_feat_percent * num_features))) {

    // This is old extraction code that would extract from the whole image
    // This can be slow as this will recompute extractions for grid areas that we have max features already
    // std::vector<cv::KeyPoint> pts1_ext;
    // Grider_FAST::perform_griding(img1pyr.at(0), mask1_updated, pts1_ext, num_features, grid_x, grid_y, threshold, true);

    // We also check a downsampled mask such that we don't extract in areas where it is all masked!
    cv::Mat mask1_grid;
    cv::resize(mask1, mask1_grid, size_grid1, 0.0, 0.0, cv::INTER_NEAREST);

    // Create grids we need to extract from and then extract our features (use fast with griding)
    int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
    int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
    std::vector<std::pair<int, int>> valid_locs;
    for (int x = 0; x < grid_2d_grid1.cols; x++) {
      for (int y = 0; y < grid_2d_grid1.rows; y++) {
        if ((int)grid_2d_grid1.at<uint8_t>(y, x) < num_features_grid_req && (int)mask1_grid.at<uint8_t>(y, x) != 255) {
          valid_locs.emplace_back(x, y);
        }
      }
    }
    std::vector<cv::KeyPoint> pts1_ext;
    if (score1)
      LearnedFeatures::detect(*score1, mask1_updated, valid_locs, pts1_ext, num_features_grid, grid_x, grid_y, letnet_minimum_score);
    else
      Grider_GRID::perform_griding(img1pyr.at(0), mask1_updated, valid_locs, pts1_ext, num_features, grid_x, grid_y, threshold, true);

    // Now, reject features that are close a current feature
    for (auto &kpt : pts1_ext) {
      // Check that it is in bounds
      int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
      int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
      if (x_grid < 0 || x_grid >= size_close1.width || y_grid < 0 || y_grid >= size_close1.height)
        continue;
      // See if there is a point at this location
      if (grid_2d_close1.at<uint8_t>(y_grid, x_grid) > 127)
        continue;
      // Else lets add it!
      pts1.push_back(kpt);
      size_t temp = ++currid;
      ids1.push_back(temp);
      grid_2d_close1.at<uint8_t>(y_grid, x_grid) = 255;
    }
  }
}

void TrackKLT::perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &kpts0,
                                std::vector<cv::KeyPoint> &kpts1, size_t id0, size_t id1, std::vector<uchar> &mask_out,
                                std::vector<FlowQuality> *diagnostics, const std::vector<size_t> *feature_ids) {

  // We must have equal vectors
  assert(kpts0.size() == kpts1.size());
  mask_out.assign(kpts0.size(), 0);
  if (diagnostics)
    diagnostics->assign(kpts0.size(), FlowQuality());

  // Return if we don't have any points
  if (kpts0.empty() || kpts1.empty())
    return;

  // Convert keypoints into points (stupid opencv stuff)
  std::vector<cv::Point2f> pts0, pts1;
  for (size_t i = 0; i < kpts0.size(); i++) {
    pts0.push_back(kpts0.at(i).pt);
    pts1.push_back(kpts1.at(i).pt);
  }

  // If we don't have enough points for ransac just return empty
  // We set the mask to be all zeros since all points failed RANSAC
  if (pts0.size() < 10) {
    return;
  }

  // Now do KLT tracking to get the valid new points
  std::vector<uchar> mask_klt;
  std::vector<float> error;
  cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
  const bool learned = neural_flow && id0 == id1;
  if (learned) {
    if (!feature_ids || feature_ids->size() != pts0.size())
      throw std::invalid_argument("Learned temporal tracking requires stable feature IDs");
    const auto prediction = neural_flow->predict(id0, img0pyr[0], img1pyr[0], pts0, *feature_ids);
    pts1 = prediction.points;
    mask_klt.resize(pts0.size());
    error.assign(pts0.size(), 0.0f);
    for (size_t i = 0; i < pts0.size(); ++i) {
      const auto &point = pts1[i];
      mask_klt[i] = std::isfinite(prediction.confidence[i]) && prediction.confidence[i] >= neural_min_confidence &&
                    std::isfinite(point.x) && std::isfinite(point.y) && point.x >= 0 && point.y >= 0 &&
                    point.x < img1pyr[0].cols && point.y < img1pyr[0].rows;
      if (!mask_klt[i]) {
        pts1[i] = pts0[i]; // Keep rejected coordinates finite for downstream image-mask indexing.
      } else if (diagnostics) {
        cv::Mat before, after;
        cv::getRectSubPix(img0pyr[0], win_size, pts0[i], before, CV_32F);
        cv::getRectSubPix(img1pyr[0], win_size, pts1[i], after, CV_32F);
        error[i] = static_cast<float>(cv::norm(before, after, cv::NORM_L1) / before.total());
      }
    }
  } else {
    cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0, pts1, mask_klt, error, win_size, pyr_levels, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);
  }

  if (diagnostics) {
    std::vector<std::pair<float, size_t>> candidates;
    for (size_t i = 0; i < pts0.size(); ++i) {
      (*diagnostics)[i].temporal = error[i];
      if (mask_klt[i] && std::isfinite(error[i]) && std::isfinite(pts1[i].x) && std::isfinite(pts1[i].y) &&
          pts1[i].x >= 0 && pts1[i].y >= 0 && pts1[i].x < img1pyr[0].cols && pts1[i].y < img1pyr[0].rows)
        candidates.emplace_back(error[i], i);
    }
    // Bounded selective round-trip diagnostics; unchecked tracks keep NaN (neutral), not a fictitious zero error.
    std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    const size_t budget = std::min(candidates.size(), std::min<size_t>(8, std::max<size_t>(3, candidates.size() / 10)));
    if (budget > 0) {
      std::vector<cv::Point2f> forward, backward;
      for (size_t i = 0; i < budget; ++i) {
        forward.push_back(pts1[candidates[i].second]);
        backward.push_back(pts0[candidates[i].second]);
      }
      std::vector<uchar> reverse_status;
      std::vector<float> reverse_error;
      cv::calcOpticalFlowPyrLK(img1pyr, img0pyr, forward, backward, reverse_status, reverse_error, win_size, pyr_levels,
                               term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);
      for (size_t i = 0; i < budget; ++i) {
        const double distance = cv::norm(backward[i] - pts0[candidates[i].second]);
        (*diagnostics)[candidates[i].second].backward = reverse_status[i] && std::isfinite(distance) ? distance : 20.0;
      }
    }
  }

  // A one-way LK success is insufficient for occlusions and repeated texture. When enabled,
  // require the target patch to track back to the original point before fitting the geometric model.
  if (max_fb_error > 0.0) {
    std::vector<cv::Point2f> forward, backward;
    std::vector<size_t> indices;
    for (size_t i = 0; i < pts0.size(); ++i) {
      const auto &p = pts1[i];
      if (!mask_klt[i] || !std::isfinite(p.x) || !std::isfinite(p.y) || p.x < 0 || p.y < 0 || p.x >= img1pyr[0].cols ||
          p.y >= img1pyr[0].rows) {
        mask_klt[i] = 0;
        continue;
      }
      forward.push_back(p);
      backward.push_back(pts0[i]);
      indices.push_back(i);
    }
    if (!forward.empty()) {
      std::vector<uchar> reverse_status;
      cv::calcOpticalFlowPyrLK(img1pyr, img0pyr, forward, backward, reverse_status, error, win_size, pyr_levels, term_crit,
                               cv::OPTFLOW_USE_INITIAL_FLOW);
      const double threshold_squared = max_fb_error * max_fb_error;
      for (size_t i = 0; i < indices.size(); ++i) {
        const auto delta = backward[i] - pts0[indices[i]];
        mask_klt[indices[i]] =
            reverse_status[i] && std::isfinite(delta.x) && std::isfinite(delta.y) && delta.dot(delta) <= threshold_squared;
      }
    }
  }

  mask_out = geometric_inlier_mask(pts0, pts1, mask_klt, id0, id1, learned || ransac_valid_only || max_fb_error > 0.0);

  // Copy back the updated positions
  for (size_t i = 0; i < pts0.size(); i++) {
    kpts0.at(i).pt = pts0.at(i);
    kpts1.at(i).pt = pts1.at(i);
  }
}

std::vector<uchar> TrackKLT::geometric_inlier_mask(const std::vector<cv::Point2f> &pts0, const std::vector<cv::Point2f> &pts1,
                                                   const std::vector<uchar> &flow_status, size_t id0, size_t id1, bool valid_only) {
  assert(pts0.size() == pts1.size() && pts0.size() == flow_status.size());
  std::vector<uchar> result(pts0.size(), 0);
  std::vector<cv::Point2f> valid0, valid1;
  std::vector<size_t> indices;
  const auto camera0 = camera_calib.at(id0), camera1 = camera_calib.at(id1);
  const auto inside = [](const cv::Point2f &p, const std::shared_ptr<CamBase> &camera) {
    return std::isfinite(p.x) && std::isfinite(p.y) && p.x >= 0 && p.y >= 0 && p.x < camera->w() && p.y < camera->h();
  };
  for (size_t i = 0; i < pts0.size(); ++i) {
    if (valid_only && (!flow_status[i] || !inside(pts0[i], camera0) || !inside(pts1[i], camera1)))
      continue;
    valid0.push_back(pts0[i]);
    valid1.push_back(pts1[i]);
    indices.push_back(i);
  }
  if (indices.size() < 10)
    return result;

  // In valid-only mode, failed LK predictions do not influence RANSAC's model/inlier ratio.
  // The default mode preserves the original finite-input tracking behavior for accuracy ablations.
  auto normalized0 = camera0->undistort_cv(valid0);
  auto normalized1 = camera1->undistort_cv(valid1);
  valid0.clear();
  valid1.clear();
  std::vector<size_t> finite_indices;
  for (size_t i = 0; i < indices.size(); ++i) {
    if (!std::isfinite(normalized0[i].x) || !std::isfinite(normalized0[i].y) || !std::isfinite(normalized1[i].x) ||
        !std::isfinite(normalized1[i].y))
      continue;
    valid0.push_back(normalized0[i]);
    valid1.push_back(normalized1[i]);
    finite_indices.push_back(indices[i]);
  }
  if (finite_indices.size() < 10)
    return result;

  const double max_focal = std::max({camera0->get_K()(0, 0), camera0->get_K()(1, 1), camera1->get_K()(0, 0), camera1->get_K()(1, 1)});
  std::vector<uchar> geometric_status;
  cv::findFundamentalMat(valid0, valid1, cv::FM_RANSAC, 2.0 / max_focal, 0.999, geometric_status);
  for (size_t i = 0; i < geometric_status.size(); ++i)
    result[finite_indices[i]] = flow_status[finite_indices[i]] && geometric_status[i];
  return result;
}
