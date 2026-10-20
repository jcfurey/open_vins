/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef OV_CORE_STEREO_KLT_H
#define OV_CORE_STEREO_KLT_H

#include "cam/CamBase.h"
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ov_core {

/// Optional validation of shared stereo observations, independent of temporal RANSAC.
class StereoKLT {
public:
  struct Statistics { size_t checked=0,accepted=0,refined=0,rejected=0; };
  using Flow=std::function<void(const std::vector<cv::Point2f> &,std::vector<cv::Point2f> &,
                                std::vector<uchar> &,bool reverse)>;
  StereoKLT(std::unordered_map<size_t,std::shared_ptr<CamBase>> cameras,const std::string &mode,
            const std::map<size_t,Eigen::VectorXd> &extrinsics,double epipolar=1.5,double backward=.75,double refinement=2);
  void update_extrinsics(const std::map<size_t,Eigen::VectorXd> &extrinsics);
  /// Fresh pairs already have a forward LK solution. Existing pairs are independently tracked temporal predictions.
  /// Rejected pairs keep their original coordinates; only successful refinement commits a new right pixel.
  void check(size_t left_camera,size_t right_camera,const std::vector<cv::Point2f> &left,
             std::vector<cv::Point2f> &right,std::vector<uchar> &status,const Flow &flow,bool fresh);
  Statistics statistics() const {return stats;}
  /// RMS symmetric epipolar distance, using distortion Jacobians to express both distances in raw pixels.
  static double epipolar_distance(CamBase &left_camera,CamBase &right_camera,const Eigen::Matrix3d &essential,
                                  const cv::Point2f &left,const cv::Point2f &right);
private:
  std::unordered_map<size_t,std::shared_ptr<CamBase>> cameras;
  std::map<size_t,Eigen::Matrix4d> transforms;
  bool refine;
  double epipolar,backward,refinement;
  Statistics stats;
};
} // namespace ov_core
#endif
