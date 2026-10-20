/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <gtest/gtest.h>
#include <limits>
#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"
#include "track/StereoKLT.h"
#include "track/TrackKLT.h"
#include "utils/quat_ops.h"

namespace {
std::shared_ptr<ov_core::CamBase> camera(bool fisheye=false,bool distorted=false) {
  std::shared_ptr<ov_core::CamBase> result;
  if(fisheye)result=std::make_shared<ov_core::CamEqui>(320,240);
  else result=std::make_shared<ov_core::CamRadtan>(320,240);
  Eigen::Matrix<double,8,1> k;k<<220,220,160,120,0,0,0,0;
  if(distorted)k.tail<4>()<<-.1,.015,.001,-.001;
  result->set_value(k);return result;
}
auto cameras() {return std::unordered_map<size_t,std::shared_ptr<ov_core::CamBase>>{{0,camera()},{1,camera()}};}
std::map<size_t,Eigen::VectorXd> extrinsics() {
  Eigen::VectorXd a(7);a<<0,0,0,1,0,0,0;auto b=a;b(4)=-.1;return {{0,a},{1,b}};
}
cv::Point2f project(ov_core::CamBase &c,const Eigen::Vector3d &p) {
  const auto q=c.distort_d(p.head<2>()/p.z());return cv::Point2f(q.x(),q.y());
}
}

TEST(StereoKlt, GeometryUsesCalibratedRotationAndDistortedPixelJacobians) {
  const Eigen::Matrix3d R=ov_core::exp_so3(Eigen::Vector3d(.03,-.02,.01));
  const Eigen::Vector3d t(-.1,.008,.003),p(.8,-.3,3);
  const Eigen::Matrix3d E=ov_core::skew_x(t)*R;
  for(bool fisheye:{false,true}) {
    auto l=camera(fisheye,true),r=camera(fisheye,true);const auto a=project(*l,p),b=project(*r,R*p+t);
    EXPECT_LT(ov_core::StereoKLT::epipolar_distance(*l,*r,E,a,b),.002);
    EXPECT_GT(ov_core::StereoKLT::epipolar_distance(*l,*r,E,a,b+cv::Point2f(0,3)),2);
    EXPECT_TRUE(std::isinf(ov_core::StereoKLT::epipolar_distance(*l,*r,Eigen::Matrix3d::Zero(),a,b)));
  }
  auto l=camera(),r=camera();const Eigen::Matrix3d rectified=ov_core::skew_x(Eigen::Vector3d(-.1,0,0));
  EXPECT_NEAR(ov_core::StereoKLT::epipolar_distance(*l,*r,rectified,{140,100},{134,101.2f}),1.2,.0001);
}

TEST(StereoKlt, EpipolarGateRejectsWrongPairsAndPreservesFailedCoordinates) {
  ov_core::StereoKLT check(cameras(),"epipolar",extrinsics());
  std::vector<cv::Point2f> left{{140,100},{180,140},{200,80},{100,90}},right{{134,100},{174,143},{194,80},{94,90}};
  const auto original=right;std::vector<uchar> status{1,1,0,1};right[3].x=std::numeric_limits<float>::quiet_NaN();
  check.check(0,1,left,right,status,{},false);
  EXPECT_EQ(status,(std::vector<uchar>{1,0,0,0}));EXPECT_EQ(right[0],original[0]);EXPECT_EQ(right[1],original[1]);
  EXPECT_EQ(check.statistics().checked,3u);EXPECT_EQ(check.statistics().accepted,1u);EXPECT_EQ(check.statistics().rejected,2u);
}

TEST(StereoKlt, RefinementCommitsOnlyRoundTripAndGeometryConsistentPixels) {
  ov_core::StereoKLT check(cameras(),"refine",extrinsics());
  const std::vector<cv::Point2f> left{{140,100},{180,140},{200,80}};
  std::vector<cv::Point2f> right{{135,100.6f},{175,140.6f},{195,80.6f}};const auto original=right;
  std::vector<uchar> status(3,1);int calls=0;
  check.check(0,1,left,right,status,[&](const auto &from,auto &to,auto &mask,bool reverse) {
    ++calls;mask.assign(from.size(),1);
    if(!reverse) {
      for(size_t i=0;i<to.size();++i)to[i]=left[i]-cv::Point2f(6,0);
      mask[2]=0;to[2].x=std::numeric_limits<float>::quiet_NaN();
    } else {
      ASSERT_EQ(from.size(),2u);to[0]=left[0];to[1]=left[1]+cv::Point2f(1,0);
    }
  },false);
  EXPECT_EQ(calls,2);EXPECT_EQ(status,(std::vector<uchar>{1,0,0}));
  EXPECT_EQ(right[0],left[0]-cv::Point2f(6,0));EXPECT_EQ(right[1],original[1]);EXPECT_EQ(right[2],original[2]);
  EXPECT_EQ(check.statistics().refined,1u);
}

TEST(StereoKlt, FreshPairsReuseForwardFlowAndBoundExistingRefinement) {
  const std::vector<cv::Point2f> left{{140,100}};
  ov_core::StereoKLT fresh(cameras(),"refine",extrinsics());std::vector<cv::Point2f> right{{134,100}};std::vector<uchar> status{1};
  int calls=0;
  fresh.check(0,1,left,right,status,[&](const auto &from,auto &to,auto &mask,bool reverse) {
    ++calls;EXPECT_TRUE(reverse);EXPECT_EQ(from.size(),1u);mask={1};to=left;
  },true);
  EXPECT_EQ(calls,1);EXPECT_EQ(status[0],1);EXPECT_EQ(fresh.statistics().refined,0u);
  ov_core::StereoKLT bound(cameras(),"refine",extrinsics());const auto original=right;
  bound.check(0,1,left,right,status,[&](const auto &,auto &to,auto &mask,bool reverse) {
    mask={1};to={reverse?left[0]:cv::Point2f(130,100)};
  },false);
  EXPECT_EQ(status[0],0);EXPECT_EQ(right,original);
}

TEST(StereoKlt, CalibrationAndVectorFailuresAreExplicitAndTransactional) {
  auto ext=extrinsics();ov_core::StereoKLT check(cameras(),"epipolar",ext);
  auto bad=ext;bad[1](3)=0;EXPECT_THROW(check.update_extrinsics(bad),std::invalid_argument);
  std::vector<cv::Point2f> left{{140,100}},right{{134,100}};std::vector<uchar> status{1};
  check.check(0,1,left,right,status,{},false);EXPECT_EQ(status[0],1);
  ext[1](5)=.1;check.update_extrinsics(ext);check.check(0,1,left,right,status,{},false);EXPECT_EQ(status[0],0);
  right.clear();EXPECT_THROW(check.check(0,1,left,right,status,{},false),std::invalid_argument);
  EXPECT_THROW(ov_core::StereoKLT(cameras(),"bad",extrinsics()),std::invalid_argument);
  bad=extrinsics();bad[1]=bad[0];EXPECT_THROW(ov_core::StereoKLT(cameras(),"refine",bad),std::invalid_argument);
}

TEST(StereoKlt, StereoFeedsRejectDriftingSharedPairsBeforeDatabaseCommit) {
  ov_core::TrackKLT tracker(cameras(),120,0,true,ov_core::TrackBase::NONE,10,4,3,8,3,15);
  tracker.set_stereo_consistency("refine",extrinsics());
  cv::Mat base(240,320,CV_8UC1);cv::RNG rng(123);rng.fill(base,cv::RNG::UNIFORM,30,220);cv::GaussianBlur(base,base,{3,3},.6);
  auto shifted=[&](float x,float y) {
    cv::Mat result,affine=(cv::Mat_<float>(2,3)<<1,0,x,0,1,y);cv::warpAffine(base,result,affine,base.size());return result;
  };
  for(int frame=0;frame<7;++frame) {
    ov_core::CameraData data;data.timestamp=1+.05*frame;data.sensor_ids={0,1};
    data.images={shifted(.3f*frame,0),shifted(-4+.3f*frame,frame==6?4.f:0.f)};
    data.masks={cv::Mat::zeros(base.size(),CV_8UC1),cv::Mat::zeros(base.size(),CV_8UC1)};
    tracker.feed_new_camera(data);
    const auto ids=tracker.get_last_ids();const auto pixels=tracker.get_last_obs();
    std::map<size_t,cv::Point2f> right;
    for(size_t i=0;i<ids.at(1).size();++i)right[ids.at(1)[i]]=pixels.at(1)[i].pt;
    size_t shared=0;
    for(size_t i=0;i<ids.at(0).size();++i) {
      auto found=right.find(ids.at(0)[i]);if(found==right.end())continue;++shared;
      EXPECT_LE(std::abs(pixels.at(0)[i].pt.y-found->second.y),1.6);
    }
    if(frame==5) {EXPECT_GT(shared,40u);}
    if(frame==6) {EXPECT_LT(shared,10u);}
  }
  EXPECT_GT(tracker.stereo_statistics().refined,100u);EXPECT_GT(tracker.stereo_statistics().rejected,40u);
}

TEST(StereoKlt, InactiveAssistanceLeavesInitializationPixelsAndIdsUnchanged) {
  ov_core::TrackKLT native(cameras(),100,0,true,ov_core::TrackBase::NONE,10,4,3,8,3,15);
  ov_core::TrackKLT configured(cameras(),100,0,true,ov_core::TrackBase::NONE,10,4,3,8,3,15);
  configured.set_stereo_consistency("refine",extrinsics());configured.set_stereo_active(false);
  cv::Mat base(240,320,CV_8UC1);cv::RNG rng(45);rng.fill(base,cv::RNG::UNIFORM,30,220);
  cv::GaussianBlur(base,base,{3,3},.6);
  for(int frame=0;frame<5;++frame) {
    ov_core::CameraData data;data.timestamp=1+.05*frame;data.sensor_ids={0,1};
    cv::Mat left,right,warp=(cv::Mat_<float>(2,3)<<1,0,.2f*frame,0,1,0);
    cv::warpAffine(base,left,warp,base.size());warp.at<float>(0,2)-=4;cv::warpAffine(base,right,warp,base.size());
    data.images={left,right};data.masks={cv::Mat::zeros(base.size(),CV_8UC1),cv::Mat::zeros(base.size(),CV_8UC1)};
    native.feed_new_camera(data);configured.feed_new_camera(data);
    EXPECT_EQ(native.get_last_ids(),configured.get_last_ids());
    for(size_t camera:{0u,1u}) {
      const auto a=native.get_last_obs().at(camera),b=configured.get_last_obs().at(camera);
      ASSERT_EQ(a.size(),b.size());for(size_t i=0;i<a.size();++i)EXPECT_EQ(a[i].pt,b[i].pt);
    }
  }
  EXPECT_EQ(configured.stereo_statistics().checked,0u);
}
