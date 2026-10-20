/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "StereoKLT.h"
#include "utils/quat_ops.h"
#include <cmath>
#include <limits>
#include <iterator>
#include <stdexcept>

namespace ov_core {
namespace {
bool inside(const cv::Point2f &p,CamBase &camera) {
  return std::isfinite(p.x)&&std::isfinite(p.y)&&p.x>=0&&p.y>=0&&p.x<camera.w()&&p.y<camera.h();
}
}
StereoKLT::StereoKLT(std::unordered_map<size_t,std::shared_ptr<CamBase>> cameras,const std::string &mode,
                     const std::map<size_t,Eigen::VectorXd> &extrinsics,double epipolar,double backward,double refinement)
    :cameras(std::move(cameras)),refine(mode=="refine"),epipolar(epipolar),backward(backward),refinement(refinement) {
  if ((mode!="epipolar"&&mode!="refine") || this->cameras.size()<2 ||
      !std::isfinite(epipolar)||epipolar<=0||!std::isfinite(backward)||backward<=0||!std::isfinite(refinement)||refinement<=0)
    throw std::invalid_argument("Invalid stereo KLT mode/cameras/pixel thresholds");
  update_extrinsics(extrinsics);
}
void StereoKLT::update_extrinsics(const std::map<size_t,Eigen::VectorXd> &extrinsics) {
  std::map<size_t,Eigen::Matrix4d> next;
  for(const auto &camera:cameras) {
    const auto found=extrinsics.find(camera.first);
    if(!camera.second||found==extrinsics.end()||found->second.size()!=7||!found->second.allFinite()||
       std::abs(found->second.head<4>().norm()-1)>1e-5)
      throw std::invalid_argument("Stereo KLT requires a normalized IMU-to-camera pose for every camera");
    auto T=Eigen::Matrix4d::Identity().eval();
    T.topLeftCorner<3,3>()=quat_2_Rot(found->second.head<4>());T.topRightCorner<3,1>()=found->second.tail<3>();
    next[camera.first]=T;
  }
  for(auto a=next.begin();a!=next.end();++a) for(auto b=std::next(a);b!=next.end();++b)
    if((b->second*a->second.inverse()).topRightCorner<3,1>().norm()<1e-6)
      throw std::invalid_argument("Stereo KLT requires a nonzero calibrated stereo baseline");
  transforms.swap(next);
}
double StereoKLT::epipolar_distance(CamBase &left_camera,CamBase &right_camera,const Eigen::Matrix3d &E,
                                    const cv::Point2f &left,const cv::Point2f &right) {
  const double invalid=std::numeric_limits<double>::infinity();
  if(!inside(left,left_camera)||!inside(right,right_camera)||!E.allFinite())return invalid;
  const Eigen::Vector2d l=left_camera.undistort_d(Eigen::Vector2d(left.x,left.y));
  const Eigen::Vector2d r=right_camera.undistort_d(Eigen::Vector2d(right.x,right.y));
  if(!l.allFinite()||!r.allFinite())return invalid;
  const Eigen::Vector3d lb(l.x(),l.y(),1),rb(r.x(),r.y(),1),ll=E.transpose()*rb,lr=E*lb;
  Eigen::MatrixXd dl,dr,unused;
  left_camera.compute_distort_jacobian(l,dl,unused);right_camera.compute_distort_jacobian(r,dr,unused);
  if(!dl.allFinite()||!dr.allFinite()||std::abs(dl.determinant())<1e-9||std::abs(dr.determinant())<1e-9)return invalid;
  const Eigen::Vector2d gl=dl.topLeftCorner<2,2>().inverse().transpose()*ll.head<2>();
  const Eigen::Vector2d gr=dr.topLeftCorner<2,2>().inverse().transpose()*lr.head<2>();
  const double a=gl.squaredNorm(),b=gr.squaredNorm(),residual=rb.dot(lr);
  if(!std::isfinite(a)||!std::isfinite(b)||a<1e-20||b<1e-20)return invalid;
  return std::abs(residual)*std::sqrt(.5*(1/a+1/b));
}
void StereoKLT::check(size_t left_camera,size_t right_camera,const std::vector<cv::Point2f> &left,
                      std::vector<cv::Point2f> &right,std::vector<uchar> &status,const Flow &flow,bool fresh) {
  if(left.size()!=right.size()||status.size()!=left.size()||left_camera==right_camera)
    throw std::invalid_argument("Stereo KLT requires corresponding pixel/status vectors from distinct cameras");
  auto &lc=*cameras.at(left_camera);auto &rc=*cameras.at(right_camera);
  const Eigen::Matrix4d T=transforms.at(right_camera)*transforms.at(left_camera).inverse();
  const Eigen::Matrix3d E=skew_x(T.topRightCorner<3,1>())*T.topLeftCorner<3,3>();
  std::vector<size_t> indices;
  std::vector<cv::Point2f> lp,rp;
  for(size_t i=0;i<left.size();++i) if(status[i]) {
    ++stats.checked;
    if(!inside(left[i],lc)||!inside(right[i],rc)) {status[i]=0;++stats.rejected;continue;}
    indices.push_back(i);lp.push_back(left[i]);rp.push_back(right[i]);
  }
  std::vector<uchar> forward(indices.size(),1),reverse(indices.size(),0);
  std::vector<cv::Point2f> back=lp;
  if(refine&&!indices.empty()) {
    if(!flow)throw std::invalid_argument("Stereo refinement requires an optical-flow callback");
    if(!fresh)flow(lp,rp,forward,false);
    if(rp.size()!=indices.size()||forward.size()!=indices.size())throw std::invalid_argument("Invalid stereo forward flow result");
    std::vector<size_t> valid;
    std::vector<cv::Point2f> from,to;
    for(size_t j=0;j<indices.size();++j) if(forward[j]&&inside(rp[j],rc)) {
      valid.push_back(j);from.push_back(rp[j]);to.push_back(lp[j]);
    }
    if(!valid.empty()) {
      std::vector<uchar> mask;
      flow(from,to,mask,true);
      if(to.size()!=valid.size()||mask.size()!=valid.size())throw std::invalid_argument("Invalid stereo reverse flow result");
      for(size_t k=0;k<valid.size();++k) {back[valid[k]]=to[k];reverse[valid[k]]=mask[k];}
    }
  }
  for(size_t j=0;j<indices.size();++j) {
    const size_t i=indices[j];
    const bool matched=!refine||(forward[j]&&reverse[j]&&inside(back[j],lc)&&cv::norm(back[j]-lp[j])<=backward&&
                                    (fresh||cv::norm(rp[j]-right[i])<=refinement));
    if(matched&&epipolar_distance(lc,rc,E,lp[j],rp[j])<=epipolar) {
      ++stats.accepted;
      if(refine&&!fresh) {right[i]=rp[j];++stats.refined;}
    } else {status[i]=0;++stats.rejected;}
  }
}
} // namespace ov_core
