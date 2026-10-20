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

#ifndef OV_MSCKF_VIOMANAGEROPTIONS_H
#define OV_MSCKF_VIOMANAGEROPTIONS_H

#include <Eigen/Eigen>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

#include "state/StateOptions.h"
#include "update/UpdaterOptions.h"
#include "utils/NoiseManager.h"

#include "init/InertialInitializerOptions.h"

#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"
#include "feat/FeatureInitializerOptions.h"
#include "track/TrackBase.h"
#include "utils/colors.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

namespace ov_msckf {

/**
 * @brief Struct which stores all options needed for state estimation.
 *
 * This is broken into a few different parts: estimator, trackers, and simulation.
 * If you are going to add a parameter here you will need to add it to the parsers.
 * You will also need to add it to the print statement at the bottom of each.
 */
struct VioManagerOptions {

  /**
   * @brief This function will load the non-simulation parameters of the system and print.
   * @param parser If not null, this parser will be used to load our parameters
   */
  void print_and_load(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
    print_and_load_estimator(parser);
    print_and_load_trackers(parser);
    print_and_load_noise(parser);

    // needs to be called last
    print_and_load_state(parser);
  }

  // ESTIMATOR ===============================

  /// Core state options (e.g. number of cameras, use fej, stereo, what calibration to enable etc)
  StateOptions state_options;

  /// Our state initialization options (e.g. window size, num features, if we should get the calibration)
  ov_init::InertialInitializerOptions init_options;

  /// Delay, in seconds, that we should wait from init before we start estimating SLAM features
  double dt_slam_delay = 2.0;

  /// If we should try to use zero velocity update
  bool try_zupt = false;

  /// Max velocity we will consider to try to do a zupt (i.e. if above this, don't do zupt)
  double zupt_max_velocity = 1.0;

  /// Multiplier of our zupt measurement IMU noise matrix (default should be 1.0)
  double zupt_noise_multiplier = 1.0;

  /// Max disparity we will consider to try to do a zupt (i.e. if above this, don't do zupt)
  double zupt_max_disparity = 1.0;

  /// If we should only use the zupt at the very beginning static initialization phase
  bool zupt_only_at_beginning = false;

  /// If we should record the timing performance to file
  bool record_timing_information = false;

  /// The path to the file we will record the timing information into
  std::string record_timing_filepath = "ov_msckf_timing.txt";

  /**
   * @brief This function will load print out all estimator settings loaded.
   * This allows for visual checking that everything was loaded properly from ROS/CMD parsers.
   *
   * @param parser If not null, this parser will be used to load our parameters
   */
  void print_and_load_estimator(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
    PRINT_DEBUG("ESTIMATOR PARAMETERS:\n");
    state_options.print(parser);
    init_options.print_and_load(parser);
    if (parser != nullptr) {
      parser->parse_config("dt_slam_delay", dt_slam_delay);
      parser->parse_config("try_zupt", try_zupt);
      parser->parse_config("zupt_max_velocity", zupt_max_velocity);
      parser->parse_config("zupt_noise_multiplier", zupt_noise_multiplier);
      parser->parse_config("zupt_max_disparity", zupt_max_disparity);
      parser->parse_config("zupt_only_at_beginning", zupt_only_at_beginning);
      parser->parse_config("record_timing_information", record_timing_information);
      parser->parse_config("record_timing_filepath", record_timing_filepath);
    }
    PRINT_DEBUG("  - dt_slam_delay: %.1f\n", dt_slam_delay);
    PRINT_DEBUG("  - zero_velocity_update: %d\n", try_zupt);
    PRINT_DEBUG("  - zupt_max_velocity: %.2f\n", zupt_max_velocity);
    PRINT_DEBUG("  - zupt_noise_multiplier: %.2f\n", zupt_noise_multiplier);
    PRINT_DEBUG("  - zupt_max_disparity: %.4f\n", zupt_max_disparity);
    PRINT_DEBUG("  - zupt_only_at_beginning?: %d\n", zupt_only_at_beginning);
    PRINT_DEBUG("  - record timing?: %d\n", (int)record_timing_information);
    PRINT_DEBUG("  - record timing filepath: %s\n", record_timing_filepath.c_str());
  }

  // NOISE / CHI2 ============================

  /// Continuous-time IMU noise (gyroscope and accelerometer)
  NoiseManager imu_noises;

  /// Update options for MSCKF features (pixel noise and chi2 multiplier)
  UpdaterOptions msckf_options;

  /// Update options for SLAM features (pixel noise and chi2 multiplier)
  UpdaterOptions slam_options;

  /// Update options for ARUCO features (pixel noise and chi2 multiplier)
  UpdaterOptions aruco_options;

  /// Update options for zero velocity (chi2 multiplier)
  UpdaterOptions zupt_options;

  /// Optional visual measurement update implementations.
  std::string visual_update_backend = "standard";

  void configure_visual_update() {
    const bool rdic = visual_update_backend == "rdic" || visual_update_backend == "rdic_fej2";
    const bool fej2 = visual_update_backend == "fej2" || visual_update_backend == "rdic_fej2";
    if (visual_update_backend != "standard" && !rdic && !fej2)
      throw std::invalid_argument("visual_update_backend must be standard, rdic, fej2 or rdic_fej2");
    if (fej2 && !state_options.do_fej)
      throw std::invalid_argument("FEJ2 requires use_fej=true");
    if (rdic && !use_klt)
      throw std::invalid_argument("RDIC observation quality requires the KLT tracking frontend");
    if (!std::isfinite(msckf_options.innovation_max_variance) || msckf_options.innovation_max_variance < 1 ||
        !(msckf_options.innovation_quantile > 0 && msckf_options.innovation_quantile < 1) ||
        msckf_options.innovation_min_blocks < 1 ||
        !(slam_options.fej2_rank_tolerance > 0 && slam_options.fej2_rank_tolerance < 1))
      throw std::invalid_argument("Invalid visual update backend settings");
    msckf_options.use_observation_covariance = slam_options.use_observation_covariance = rdic;
    msckf_options.calibrate_innovation = slam_options.calibrate_innovation = rdic;
    slam_options.innovation_max_variance = msckf_options.innovation_max_variance;
    slam_options.innovation_quantile = msckf_options.innovation_quantile;
    slam_options.innovation_min_blocks = msckf_options.innovation_min_blocks;
    msckf_options.use_fej2 = false;
    slam_options.use_fej2 = fej2;
  }

  /**
   * @brief This function will load print out all noise parameters loaded.
   * This allows for visual checking that everything was loaded properly from ROS/CMD parsers.
   *
   * @param parser If not null, this parser will be used to load our parameters
   */
  void print_and_load_noise(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
    PRINT_DEBUG("NOISE PARAMETERS:\n");
    if (parser != nullptr) {
      parser->parse_external("relative_config_imu", "imu0", "gyroscope_noise_density", imu_noises.sigma_w);
      parser->parse_external("relative_config_imu", "imu0", "gyroscope_random_walk", imu_noises.sigma_wb);
      parser->parse_external("relative_config_imu", "imu0", "accelerometer_noise_density", imu_noises.sigma_a);
      parser->parse_external("relative_config_imu", "imu0", "accelerometer_random_walk", imu_noises.sigma_ab);
    }
    imu_noises.print();
    if (parser != nullptr) {
      parser->parse_config("up_msckf_sigma_px", msckf_options.sigma_pix);
      parser->parse_config("up_msckf_chi2_multipler", msckf_options.chi2_multipler);
      parser->parse_config("up_slam_sigma_px", slam_options.sigma_pix);
      parser->parse_config("up_slam_chi2_multipler", slam_options.chi2_multipler);
      parser->parse_config("up_aruco_sigma_px", aruco_options.sigma_pix);
      parser->parse_config("up_aruco_chi2_multipler", aruco_options.chi2_multipler);
      msckf_options.sigma_pix_sq = std::pow(msckf_options.sigma_pix, 2);
      slam_options.sigma_pix_sq = std::pow(slam_options.sigma_pix, 2);
      aruco_options.sigma_pix_sq = std::pow(aruco_options.sigma_pix, 2);
      parser->parse_config("zupt_chi2_multipler", zupt_options.chi2_multipler);
      parser->parse_config("visual_update_backend", visual_update_backend, false);
      parser->parse_config("rdic_max_variance", msckf_options.innovation_max_variance, false);
      parser->parse_config("rdic_quantile", msckf_options.innovation_quantile, false);
      parser->parse_config("rdic_min_blocks", msckf_options.innovation_min_blocks, false);
      parser->parse_config("fej2_rank_tolerance", slam_options.fej2_rank_tolerance, false);
    }
    configure_visual_update();
    PRINT_INFO("Visual update backend: %s\n", visual_update_backend.c_str());
    PRINT_DEBUG("  Updater MSCKF Feats:\n");
    msckf_options.print();
    PRINT_DEBUG("  Updater SLAM Feats:\n");
    slam_options.print();
    PRINT_DEBUG("  Updater ARUCO Tags:\n");
    aruco_options.print();
    PRINT_DEBUG("  Updater ZUPT:\n");
    zupt_options.print();
  }

  // STATE DEFAULTS ==========================

  /// Gravity magnitude in the global frame (i.e. should be 9.81 typically)
  double gravity_mag = 9.81;

  /// Gyroscope IMU intrinsics (scale imperfection and axis misalignment, column-wise, inverse)
  Eigen::Matrix<double, 6, 1> vec_dw;

  /// Accelerometer IMU intrinsics (scale imperfection and axis misalignment, column-wise, inverse)
  Eigen::Matrix<double, 6, 1> vec_da;

  /// Gyroscope gravity sensitivity (scale imperfection and axis misalignment, column-wise)
  Eigen::Matrix<double, 9, 1> vec_tg;

  /// Rotation from gyroscope frame to the "IMU" accelerometer frame
  Eigen::Matrix<double, 4, 1> q_ACCtoIMU;

  /// Rotation from accelerometer to the "IMU" gyroscope frame frame
  Eigen::Matrix<double, 4, 1> q_GYROtoIMU;

  /// Time offset between camera and IMU.
  double calib_camimu_dt = 0.0;

  /// Map between camid and camera intrinsics (fx, fy, cx, cy, d1...d4, cam_w, cam_h)
  std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> camera_intrinsics;

  /// Map between camid and camera extrinsics (q_ItoC, p_IinC).
  std::map<size_t, Eigen::VectorXd> camera_extrinsics;

  /// If we should try to load a mask and use it to reject invalid features
  bool use_mask = false;

  /// Mask images for each camera
  std::map<size_t, cv::Mat> masks;

  /**
   * @brief Per-camera masks at the resolution of incoming images (before any downsampling).
   *
   * Configured masks match the calibration, which is already halved when downsampling, while the estimator halves
   * each incoming image and its mask together. Masks are upsampled to the input size so both stay aligned.
   * Without masks this is an all-zero mask per camera.
   */
  std::map<int, cv::Mat> input_masks() const {
    std::map<int, cv::Mat> result;
    for (const auto &camera : camera_intrinsics) {
      const int factor = downsample_cameras ? 2 : 1;
      const cv::Size size(camera.second->w() * factor, camera.second->h() * factor);
      cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
      if (use_mask) {
        const cv::Mat &configured = masks.at(camera.first);
        if (configured.size() == size)
          mask = configured;
        else
          cv::resize(configured, mask, size, 0.0, 0.0, cv::INTER_NEAREST);
      }
      result.emplace(static_cast<int>(camera.first), mask);
    }
    return result;
  }

  /**
   * @brief This function will load and print all state parameters (e.g. sensor extrinsics)
   * This allows for visual checking that everything was loaded properly from ROS/CMD parsers.
   *
   * @param parser If not null, this parser will be used to load our parameters
   */
  void print_and_load_state(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
    if (parser != nullptr) {
      parser->parse_config("gravity_mag", gravity_mag);
      for (int i = 0; i < state_options.num_cameras; i++) {

        // Time offset (use the first one)
        // TODO: support multiple time offsets between cameras
        if (i == 0) {
          parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "timeshift_cam_imu", calib_camimu_dt, false);
        }

        // Distortion model
        std::string dist_model = "radtan";
        parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "distortion_model", dist_model);

        // Distortion parameters
        std::vector<double> cam_calib1 = {1, 1, 0, 0};
        std::vector<double> cam_calib2 = {0, 0, 0, 0};
        parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "intrinsics", cam_calib1);
        parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "distortion_coeffs", cam_calib2);
        Eigen::VectorXd cam_calib = Eigen::VectorXd::Zero(8);
        cam_calib << cam_calib1.at(0), cam_calib1.at(1), cam_calib1.at(2), cam_calib1.at(3), cam_calib2.at(0), cam_calib2.at(1),
            cam_calib2.at(2), cam_calib2.at(3);
        cam_calib(0) /= (downsample_cameras) ? 2.0 : 1.0;
        cam_calib(1) /= (downsample_cameras) ? 2.0 : 1.0;
        cam_calib(2) /= (downsample_cameras) ? 2.0 : 1.0;
        cam_calib(3) /= (downsample_cameras) ? 2.0 : 1.0;

        // FOV / resolution
        std::vector<int> matrix_wh = {1, 1};
        parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "resolution", matrix_wh);
        matrix_wh.at(0) /= (downsample_cameras) ? 2.0 : 1.0;
        matrix_wh.at(1) /= (downsample_cameras) ? 2.0 : 1.0;

        // Extrinsics
        Eigen::Matrix4d T_CtoI = Eigen::Matrix4d::Identity();
        parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "T_imu_cam", T_CtoI);

        // Load these into our state
        Eigen::Matrix<double, 7, 1> cam_eigen;
        cam_eigen.block(0, 0, 4, 1) = ov_core::rot_2_quat(T_CtoI.block(0, 0, 3, 3).transpose());
        cam_eigen.block(4, 0, 3, 1) = -T_CtoI.block(0, 0, 3, 3).transpose() * T_CtoI.block(0, 3, 3, 1);

        // Create intrinsics model
        if (dist_model == "equidistant") {
          camera_intrinsics.insert({i, std::make_shared<ov_core::CamEqui>(matrix_wh.at(0), matrix_wh.at(1))});
          camera_intrinsics.at(i)->set_value(cam_calib);
        } else {
          camera_intrinsics.insert({i, std::make_shared<ov_core::CamRadtan>(matrix_wh.at(0), matrix_wh.at(1))});
          camera_intrinsics.at(i)->set_value(cam_calib);
        }
        camera_extrinsics.insert({i, cam_eigen});
      }
      parser->parse_config("use_mask", use_mask);
      if (use_mask) {
        for (int i = 0; i < state_options.num_cameras; i++) {
          std::string mask_path;
          std::string mask_node = "mask" + std::to_string(i);
          parser->parse_config(mask_node, mask_path);
          std::string total_mask_path = parser->get_config_folder() + mask_path;
          if (!boost::filesystem::exists(total_mask_path)) {
            PRINT_ERROR(RED "VioManager(): invalid mask path:\n" RESET);
            PRINT_ERROR(RED "\t- mask%d - %s\n" RESET, i, total_mask_path.c_str());
            std::exit(EXIT_FAILURE);
          }
          cv::Mat mask = cv::imread(total_mask_path, cv::IMREAD_GRAYSCALE);
          masks.insert({i, mask});
          if (mask.cols != camera_intrinsics.at(i)->w() || mask.rows != camera_intrinsics.at(i)->h()) {
            PRINT_ERROR(RED "VioManager(): mask size does not match camera!\n" RESET);
            PRINT_ERROR(RED "\t- mask%d - %s\n" RESET, i, total_mask_path.c_str());
            PRINT_ERROR(RED "\t- mask%d - %d x %d\n" RESET, mask.cols, mask.rows);
            PRINT_ERROR(RED "\t- cam%d - %d x %d\n" RESET, camera_intrinsics.at(i)->w(), camera_intrinsics.at(i)->h());
            std::exit(EXIT_FAILURE);
          }
        }
      }

      // IMU intrinsics
      Eigen::Matrix3d Tw = Eigen::Matrix3d::Identity();
      parser->parse_external("relative_config_imu", "imu0", "Tw", Tw);
      Eigen::Matrix3d Ta = Eigen::Matrix3d::Identity();
      parser->parse_external("relative_config_imu", "imu0", "Ta", Ta);
      Eigen::Matrix3d R_IMUtoACC = Eigen::Matrix3d::Identity();
      parser->parse_external("relative_config_imu", "imu0", "R_IMUtoACC", R_IMUtoACC);
      Eigen::Matrix3d R_IMUtoGYRO = Eigen::Matrix3d::Identity();
      parser->parse_external("relative_config_imu", "imu0", "R_IMUtoGYRO", R_IMUtoGYRO);
      Eigen::Matrix3d Tg = Eigen::Matrix3d::Zero();
      parser->parse_external("relative_config_imu", "imu0", "Tg", Tg);

      // Generate the parameters we need
      // TODO: error here if this returns a NaN value (i.e. invalid matrix specified)
      Eigen::Matrix3d Dw = Tw.colPivHouseholderQr().solve(Eigen::Matrix3d::Identity());
      Eigen::Matrix3d Da = Ta.colPivHouseholderQr().solve(Eigen::Matrix3d::Identity());
      Eigen::Matrix3d R_ACCtoIMU = R_IMUtoACC.transpose();
      Eigen::Matrix3d R_GYROtoIMU = R_IMUtoGYRO.transpose();
      if (std::isnan(Tw.norm()) || std::isnan(Dw.norm())) {
        std::stringstream ss;
        ss << "gyroscope has bad intrinsic values!" << std::endl;
        ss << "Tw - " << std::endl << Tw << std::endl << std::endl;
        ss << "Dw - " << std::endl << Dw << std::endl << std::endl;
        PRINT_DEBUG(RED "" RESET, ss.str().c_str());
        std::exit(EXIT_FAILURE);
      }
      if (std::isnan(Ta.norm()) || std::isnan(Da.norm())) {
        std::stringstream ss;
        ss << "accelerometer has bad intrinsic values!" << std::endl;
        ss << "Ta - " << std::endl << Ta << std::endl << std::endl;
        ss << "Da - " << std::endl << Da << std::endl << std::endl;
        PRINT_DEBUG(RED "" RESET, ss.str().c_str());
        std::exit(EXIT_FAILURE);
      }

      // kalibr model: lower triangular of the matrix and R_GYROtoI
      // rpng model: upper triangular of the matrix and R_ACCtoI
      if (state_options.imu_model == StateOptions::ImuModel::KALIBR) {
        vec_dw << Dw.block<3, 1>(0, 0), Dw.block<2, 1>(1, 1), Dw(2, 2);
        vec_da << Da.block<3, 1>(0, 0), Da.block<2, 1>(1, 1), Da(2, 2);
      } else {
        vec_dw << Dw(0, 0), Dw.block<2, 1>(0, 1), Dw.block<3, 1>(0, 2);
        vec_da << Da(0, 0), Da.block<2, 1>(0, 1), Da.block<3, 1>(0, 2);
      }
      vec_tg << Tg.block<3, 1>(0, 0), Tg.block<3, 1>(0, 1), Tg.block<3, 1>(0, 2);
      q_GYROtoIMU = ov_core::rot_2_quat(R_GYROtoIMU);
      q_ACCtoIMU = ov_core::rot_2_quat(R_ACCtoIMU);
    }
    PRINT_DEBUG("STATE PARAMETERS:\n");
    PRINT_DEBUG("  - gravity_mag: %.4f\n", gravity_mag);
    PRINT_DEBUG("  - gravity: %.3f, %.3f, %.3f\n", 0.0, 0.0, gravity_mag);
    PRINT_DEBUG("  - camera masks?: %d\n", use_mask);
    if (state_options.num_cameras != (int)camera_intrinsics.size() || state_options.num_cameras != (int)camera_extrinsics.size()) {
      PRINT_ERROR(RED "[SIM]: camera calib size does not match max cameras...\n" RESET);
      PRINT_ERROR(RED "[SIM]: got %d but expected %d max cameras (camera_intrinsics)\n" RESET, (int)camera_intrinsics.size(),
                  state_options.num_cameras);
      PRINT_ERROR(RED "[SIM]: got %d but expected %d max cameras (camera_extrinsics)\n" RESET, (int)camera_extrinsics.size(),
                  state_options.num_cameras);
      std::exit(EXIT_FAILURE);
    }
    PRINT_DEBUG("  - calib_camimu_dt: %.4f\n", calib_camimu_dt);
    PRINT_DEBUG("CAMERA PARAMETERS:\n");
    for (int n = 0; n < state_options.num_cameras; n++) {
      std::stringstream ss;
      ss << "cam_" << n << "_fisheye:" << (std::dynamic_pointer_cast<ov_core::CamEqui>(camera_intrinsics.at(n)) != nullptr) << std::endl;
      ss << "cam_" << n << "_wh:" << std::endl << camera_intrinsics.at(n)->w() << " x " << camera_intrinsics.at(n)->h() << std::endl;
      ss << "cam_" << n << "_intrinsic(0:3):" << std::endl
         << camera_intrinsics.at(n)->get_value().block(0, 0, 4, 1).transpose() << std::endl;
      ss << "cam_" << n << "_intrinsic(4:7):" << std::endl
         << camera_intrinsics.at(n)->get_value().block(4, 0, 4, 1).transpose() << std::endl;
      ss << "cam_" << n << "_extrinsic(0:3):" << std::endl << camera_extrinsics.at(n).block(0, 0, 4, 1).transpose() << std::endl;
      ss << "cam_" << n << "_extrinsic(4:6):" << std::endl << camera_extrinsics.at(n).block(4, 0, 3, 1).transpose() << std::endl;
      Eigen::Matrix4d T_CtoI = Eigen::Matrix4d::Identity();
      T_CtoI.block(0, 0, 3, 3) = ov_core::quat_2_Rot(camera_extrinsics.at(n).block(0, 0, 4, 1)).transpose();
      T_CtoI.block(0, 3, 3, 1) = -T_CtoI.block(0, 0, 3, 3) * camera_extrinsics.at(n).block(4, 0, 3, 1);
      ss << "T_C" << n << "toI:" << std::endl << T_CtoI << std::endl << std::endl;
      PRINT_DEBUG(ss.str().c_str());
    }
    PRINT_DEBUG("IMU PARAMETERS:\n");
    std::stringstream ss;
    ss << "imu model:" << ((state_options.imu_model == StateOptions::ImuModel::KALIBR) ? "kalibr" : "rpng") << std::endl;
    ss << "Dw (columnwise):" << vec_dw.transpose() << std::endl;
    ss << "Da (columnwise):" << vec_da.transpose() << std::endl;
    ss << "Tg (columnwise):" << vec_tg.transpose() << std::endl;
    ss << "q_GYROtoI: " << q_GYROtoIMU.transpose() << std::endl;
    ss << "q_ACCtoI: " << q_ACCtoIMU.transpose() << std::endl;
    PRINT_DEBUG(ss.str().c_str());
  }

  // TRACKERS ===============================

  /// If we should process two cameras are being stereo or binocular. If binocular, we do monocular feature tracking on each image.
  bool use_stereo = true;

  /// If we should use KLT tracking, or descriptor matcher
  bool use_klt = true;

  /// Native KLT/descriptor selection remains controlled by use_klt; the learned path needs a complete trained export.
  std::string tracking_backend = "native";
  std::string kltnet_model_path;
  std::string kltnet_target = "cpu";
  double kltnet_min_confidence = 0.1;
  std::string letnet_model_path;
  double letnet_min_score = 0.2;
  std::string letnet_inference_backend = "opencv";
  int letnet_inference_threads = 4;

  void configure_tracking() const {
    if (tracking_backend != "native" && tracking_backend != "kltnet_onnx" && tracking_backend != "letnet_onnx")
      throw std::invalid_argument("tracking_backend must be native, kltnet_onnx or letnet_onnx");
    if (tracking_backend == "letnet_onnx") {
      if (!use_klt || letnet_model_path.empty() || histogram_method != ov_core::TrackBase::NONE)
        throw std::invalid_argument("letnet_onnx requires use_klt=true, histogram_method=NONE and a prepared letnet_model_path");
      if (!std::isfinite(letnet_min_score) || letnet_min_score < 0 || letnet_min_score > 1)
        throw std::invalid_argument("letnet_min_score must be in [0,1]");
      if ((letnet_inference_backend != "opencv" && letnet_inference_backend != "onnxruntime") ||
          letnet_inference_threads < 1 || letnet_inference_threads > 64)
        throw std::invalid_argument("LET-Net inference backend must be opencv or onnxruntime, with 1..64 inference threads");
      if (msckf_options.use_observation_covariance)
        throw std::invalid_argument("RDIC photometric covariance currently expects native grayscale LK; select standard or fej2 with LET-Net");
    }
    if (tracking_backend == "kltnet_onnx") {
      if (!use_klt || kltnet_model_path.empty())
        throw std::invalid_argument("kltnet_onnx requires use_klt=true and kltnet_model_path pointing to a trained export");
      if (kltnet_target != "cpu" && kltnet_target != "cuda")
        throw std::invalid_argument("kltnet_target must be cpu or cuda");
      if (!std::isfinite(kltnet_min_confidence) || kltnet_min_confidence < 0)
        throw std::invalid_argument("kltnet_min_confidence must be finite and nonnegative");
    }
  }

  /// If should extract aruco tags and estimate them
  bool use_aruco = true;

  /// Will half the resolution of the aruco tag image (will be faster)
  bool downsize_aruco = true;

  /// Will half the resolution all tracking image (aruco will be 1/4 instead of halved if dowsize_aruoc also enabled)
  bool downsample_cameras = false;

  /// Threads our front-end should try to use (opencv uses this also)
  int num_opencv_threads = 4;

  /// If our ROS image publisher should be async (if sim this should be no!)
  bool use_multi_threading_pubs = true;

  /// If our ROS subscriber callbacks should be async (if sim and serial then this should be no!)
  bool use_multi_threading_subs = false;

  /// Total tracking target divided uniformly across cameras. This strongly affects tracking cost.
  int num_pts = 150;

  /// Fast extraction threshold
  int fast_threshold = 20;

  /// Number of grids we should split column-wise to do feature extraction in
  int grid_x = 5;

  /// Number of grids we should split row-wise to do feature extraction in
  int grid_y = 5;

  /// Will check after doing KLT track and remove any features closer than this
  int min_px_dist = 10;

  /// Maximum optical-flow pyramid level (levels 0 through this value are used).
  int klt_pyramid_levels = 5;

  /// Odd optical-flow patch width in processed-image pixels.
  int klt_window_size = 15;

  /// Maximum forward/backward optical-flow error in processed-image pixels; zero disables the check.
  double klt_max_fb_error = 0.0;

  /// Experimental valid-only RANSAC input filtering; reciprocal tracking also implies this check.
  bool klt_ransac_valid_only = false;

  /// What type of pre-processing histogram method should be applied to images
  ov_core::TrackBase::HistogramMethod histogram_method = ov_core::TrackBase::HistogramMethod::HISTOGRAM;

  /// KNN ration between top two descriptor matcher which is required to be a good match
  double knn_ratio = 0.85;

  /// Frequency we want to track images at (higher freq ones will be dropped)
  double track_frequency = 20.0;

  /// Parameters used by our feature initialize / triangulator
  ov_core::FeatureInitializerOptions featinit_options;

  /**
   * @brief This function will load print out all parameters related to visual tracking
   * This allows for visual checking that everything was loaded properly from ROS/CMD parsers.
   *
   * @param parser If not null, this parser will be used to load our parameters
   */
  void print_and_load_trackers(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
    if (parser != nullptr) {
      parser->parse_config("use_stereo", use_stereo);
      parser->parse_config("use_klt", use_klt);
      parser->parse_config("use_aruco", use_aruco);
      parser->parse_config("downsize_aruco", downsize_aruco);
      parser->parse_config("downsample_cameras", downsample_cameras);
      parser->parse_config("num_opencv_threads", num_opencv_threads);
      parser->parse_config("multi_threading_pubs", use_multi_threading_pubs, false);
      parser->parse_config("multi_threading_subs", use_multi_threading_subs, false);
      parser->parse_config("num_pts", num_pts);
      parser->parse_config("fast_threshold", fast_threshold);
      parser->parse_config("grid_x", grid_x);
      parser->parse_config("grid_y", grid_y);
      parser->parse_config("min_px_dist", min_px_dist);
      parser->parse_config("klt_pyramid_levels", klt_pyramid_levels, false);
      parser->parse_config("klt_window_size", klt_window_size, false);
      parser->parse_config("klt_max_fb_error", klt_max_fb_error, false);
      parser->parse_config("klt_ransac_valid_only", klt_ransac_valid_only, false);
      parser->parse_config("tracking_backend", tracking_backend, false);
      parser->parse_config("kltnet_model_path", kltnet_model_path, false);
      parser->parse_config("kltnet_target", kltnet_target, false);
      parser->parse_config("kltnet_min_confidence", kltnet_min_confidence, false);
      parser->parse_config("letnet_model_path", letnet_model_path, false);
      parser->parse_config("letnet_min_score", letnet_min_score, false);
      parser->parse_config("letnet_inference_backend", letnet_inference_backend, false);
      parser->parse_config("letnet_inference_threads", letnet_inference_threads, false);
      std::string histogram_method_str = "HISTOGRAM";
      parser->parse_config("histogram_method", histogram_method_str);
      if (histogram_method_str == "NONE") {
        histogram_method = ov_core::TrackBase::NONE;
      } else if (histogram_method_str == "HISTOGRAM") {
        histogram_method = ov_core::TrackBase::HISTOGRAM;
      } else if (histogram_method_str == "CLAHE") {
        histogram_method = ov_core::TrackBase::CLAHE;
      } else {
        printf(RED "VioManager(): invalid feature histogram specified:\n" RESET);
        printf(RED "\t- NONE\n" RESET);
        printf(RED "\t- HISTOGRAM\n" RESET);
        printf(RED "\t- CLAHE\n" RESET);
        std::exit(EXIT_FAILURE);
      }
      parser->parse_config("knn_ratio", knn_ratio);
      parser->parse_config("track_frequency", track_frequency);
    }
    configure_tracking();
    PRINT_INFO("Tracking backend: %s\n", tracking_backend.c_str());
    PRINT_DEBUG("FEATURE TRACKING PARAMETERS:\n");
    PRINT_DEBUG("  - use_stereo: %d\n", use_stereo);
    PRINT_DEBUG("  - use_klt: %d\n", use_klt);
    PRINT_DEBUG("  - use_aruco: %d\n", use_aruco);
    PRINT_DEBUG("  - downsize aruco: %d\n", downsize_aruco);
    PRINT_DEBUG("  - downsize cameras: %d\n", downsample_cameras);
    PRINT_DEBUG("  - num opencv threads: %d\n", num_opencv_threads);
    PRINT_DEBUG("  - use multi-threading pubs: %d\n", use_multi_threading_pubs);
    PRINT_DEBUG("  - use multi-threading subs: %d\n", use_multi_threading_subs);
    PRINT_DEBUG("  - num_pts: %d\n", num_pts);
    PRINT_DEBUG("  - fast threshold: %d\n", fast_threshold);
    PRINT_DEBUG("  - grid X by Y: %d by %d\n", grid_x, grid_y);
    PRINT_DEBUG("  - min px dist: %d\n", min_px_dist);
    PRINT_DEBUG("  - KLT max pyramid level: %d\n", klt_pyramid_levels);
    PRINT_DEBUG("  - KLT window width: %d\n", klt_window_size);
    PRINT_DEBUG("  - KLT maximum forward/backward error: %.3f px\n", klt_max_fb_error);
    PRINT_DEBUG("  - KLT valid-only RANSAC input: %d\n", klt_ransac_valid_only);
    PRINT_DEBUG("  - hist method: %d\n", (int)histogram_method);
    PRINT_DEBUG("  - knn ratio: %.3f\n", knn_ratio);
    PRINT_DEBUG("  - track frequency: %.1f\n", track_frequency);
    featinit_options.print(parser);
  }

  // SIMULATOR ===============================

  /// Seed for initial states (i.e. random feature 3d positions in the generated map)
  int sim_seed_state_init = 0;

  /// Seed for calibration perturbations. Change this to perturb by different random values if perturbations are enabled.
  int sim_seed_preturb = 0;

  /// Measurement noise seed. This should be incremented for each run in the Monte-Carlo simulation to generate the same true measurements,
  /// but diffferent noise values.
  int sim_seed_measurements = 0;

  /// If we should perturb the calibration that the estimator starts with
  bool sim_do_perturbation = false;

  /// Path to the trajectory we will b-spline and simulate on. Should be time(s),pos(xyz),ori(xyzw) format.
  std::string sim_traj_path = "src/open_vins/ov_data/sim/udel_gore.txt";

  /// We will start simulating after we have moved this much along the b-spline. This prevents static starts as we init from groundtruth in
  /// simulation.
  double sim_distance_threshold = 1.2;

  /// Frequency (Hz) that we will simulate our cameras
  double sim_freq_cam = 10.0;

  /// Frequency (Hz) that we will simulate our inertial measurement unit
  double sim_freq_imu = 400.0;

  /// Feature distance we generate features from (minimum)
  double sim_min_feature_gen_distance = 5;

  /// Feature distance we generate features from (maximum)
  double sim_max_feature_gen_distance = 10;

  /**
   * @brief This function will load print out all simulated parameters.
   * This allows for visual checking that everything was loaded properly from ROS/CMD parsers.
   *
   * @param parser If not null, this parser will be used to load our parameters
   */
  void print_and_load_simulation(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
    if (parser != nullptr) {
      parser->parse_config("sim_seed_state_init", sim_seed_state_init);
      parser->parse_config("sim_seed_preturb", sim_seed_preturb);
      parser->parse_config("sim_seed_measurements", sim_seed_measurements);
      parser->parse_config("sim_do_perturbation", sim_do_perturbation);
      parser->parse_config("sim_traj_path", sim_traj_path);
      parser->parse_config("sim_distance_threshold", sim_distance_threshold);
      parser->parse_config("sim_freq_cam", sim_freq_cam);
      parser->parse_config("sim_freq_imu", sim_freq_imu);
      parser->parse_config("sim_min_feature_gen_dist", sim_min_feature_gen_distance);
      parser->parse_config("sim_max_feature_gen_dist", sim_max_feature_gen_distance);
    }
    PRINT_DEBUG("SIMULATION PARAMETERS:\n");
    PRINT_WARNING(BOLDRED "  - state init seed: %d \n" RESET, sim_seed_state_init);
    PRINT_WARNING(BOLDRED "  - perturb seed: %d \n" RESET, sim_seed_preturb);
    PRINT_WARNING(BOLDRED "  - measurement seed: %d \n" RESET, sim_seed_measurements);
    PRINT_WARNING(BOLDRED "  - do perturb?: %d\n" RESET, sim_do_perturbation);
    PRINT_DEBUG("  - traj path: %s\n", sim_traj_path.c_str());
    PRINT_DEBUG("  - dist thresh: %.2f\n", sim_distance_threshold);
    PRINT_DEBUG("  - cam feq: %.2f\n", sim_freq_cam);
    PRINT_DEBUG("  - imu feq: %.2f\n", sim_freq_imu);
    PRINT_DEBUG("  - min feat dist: %.2f\n", sim_min_feature_gen_distance);
    PRINT_DEBUG("  - max feat dist: %.2f\n", sim_max_feature_gen_distance);
  }
};

} // namespace ov_msckf

#endif // OV_MSCKF_VIOMANAGEROPTIONS_H
