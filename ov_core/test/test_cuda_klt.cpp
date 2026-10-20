/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <gtest/gtest.h>
#include <future>
#include <limits>
#include "track/CudaKLT.h"
#include "track/TrackKLT.h"
#include "cam/CamRadtan.h"
#include <opencv2/video/tracking.hpp>

namespace {
#if ENABLE_CUDA_KLT
cv::Mat texture(int width=320,int height=240) {
  cv::Mat image(height+12,width+12,CV_8UC1);cv::RNG rng(42);rng.fill(image,cv::RNG::UNIFORM,0,256);
  cv::GaussianBlur(image,image,cv::Size(3,3),.6);
  return image(cv::Rect(6,6,width,height)); // Exercise host image stride and a non-contiguous ROI.
}
std::vector<cv::Point2f> points(int width,int height) {
  std::vector<cv::Point2f> result;
  for(int y=25;y<height-25;y+=15)for(int x=25;x<width-25;x+=15)result.emplace_back(x,y);
  return result;
}
cv::Mat translated(const cv::Mat &image,float x,float y) {
  cv::Mat result;cv::Mat warp=(cv::Mat_<float>(2,3)<<1,0,x,0,1,y);
  cv::warpAffine(image,result,warp,image.size(),cv::INTER_LINEAR,cv::BORDER_REFLECT_101);return result;
}
void motion(ov_core::CudaKLT &flow,int width,int height,float x,float y) {
  const auto previous=texture(width,height),current=translated(previous,x,y);
  const auto old=points(width,height);auto next=old;std::vector<uchar> status;std::vector<float> error;
  flow.calc(previous,current,old,next,status,error);
  size_t good=0;double sum=0;
  for(size_t i=0;i<old.size();++i)if(status[i]){++good;sum+=cv::norm(next[i]-old[i]-cv::Point2f(x,y));}
  EXPECT_GT(good,old.size()*9/10);EXPECT_LT(sum/std::max(size_t(1),good),.13);
}
#endif
}

TEST(CudaKlt, AvailabilityIsExplicitAndNeverSilentlySelectsCpu) {
#if ENABLE_CUDA_KLT
  if(!ov_core::CudaKLT::available())GTEST_SKIP()<<"CUDA runtime/device unavailable";
  EXPECT_FALSE(ov_core::CudaKLT::device_name().empty());
  EXPECT_EQ(ov_core::CudaKLT::runtime_version()/1000,ov_core::CudaKLT::compiled_version()/1000);
  EXPECT_THROW(ov_core::CudaKLT(3,33),std::invalid_argument);
#else
  EXPECT_FALSE(ov_core::CudaKLT::available());
  EXPECT_THROW(ov_core::CudaKLT(3,15),std::runtime_error);
#endif
}

#if ENABLE_CUDA_KLT
TEST(CudaKlt, RecoversSubpixelMotionAcrossChangingImageSizesAndStrides) {
  if(!ov_core::CudaKLT::available())GTEST_SKIP();
  ov_core::CudaKLT flow(3,15);
  motion(flow,320,240,2.35,-1.2);motion(flow,192,128,-.7,.4);motion(flow,752,480,9.2,-3.1);
}

TEST(CudaKlt, AgreesWithCpuLkOnKnownMotionIncludingInitialFlow) {
  if(!ov_core::CudaKLT::available())GTEST_SKIP();
  ov_core::CudaKLT flow(3,15);auto before=texture(),after=translated(before,5.3,-2.1);
  auto old=points(before.cols,before.rows),gpu=old,cpu=old;
  for(auto &p:gpu)p+=cv::Point2f(4,-2);
  cpu=gpu;
  std::vector<uchar> gs,cs;std::vector<float> ge,ce;
  flow.calc(before,after,old,gpu,gs,ge);
  cv::calcOpticalFlowPyrLK(before,after,old,cpu,cs,ce,cv::Size(15,15),3,
    cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,30,.01),cv::OPTFLOW_USE_INITIAL_FLOW);
  size_t n=0;double error=0;
  for(size_t i=0;i<old.size();++i)if(gs[i]&&cs[i]){++n;error+=cv::norm(gpu[i]-cpu[i]);}
  EXPECT_GT(n,old.size()*9/10);EXPECT_LT(error/std::max(size_t(1),n),.13);
}

TEST(CudaKlt, RejectsFlatAndInvalidPointsAndHandlesEmptyInput) {
  if(!ov_core::CudaKLT::available())GTEST_SKIP();
  ov_core::CudaKLT flow(3,15);auto image=cv::Mat::zeros(128,192,CV_8UC1);
  std::vector<cv::Point2f> old{{50,50},{std::numeric_limits<float>::quiet_NaN(),50}},next=old;
  std::vector<uchar> status;std::vector<float> error;
  flow.calc(image,image,old,next,status,error);EXPECT_EQ(status,(std::vector<uchar>{0,0}));
  EXPECT_TRUE(std::isfinite(next[0].x));EXPECT_TRUE(std::isfinite(next[1].x));
  old.clear();next.clear();flow.calc(image,image,old,next,status,error);EXPECT_TRUE(status.empty());
}

TEST(CudaKlt, IndependentCameraStreamsWorkConcurrently) {
  if(!ov_core::CudaKLT::available())GTEST_SKIP();
  ov_core::CudaKLT left(3,15),right(3,15);
  auto a=std::async(std::launch::async,[&]{motion(left,320,240,1.4,.3);});
  auto b=std::async(std::launch::async,[&]{motion(right,320,240,-1.2,.5);});a.get();b.get();
}

TEST(CudaKlt, StereoTrackerPreservesCorrespondencesAndSupportsGyroSeeding) {
  if(!ov_core::CudaKLT::available())GTEST_SKIP();
  std::unordered_map<size_t,std::shared_ptr<ov_core::CamBase>> cameras;
  for(size_t i=0;i<2;++i) {
    auto c=std::make_shared<ov_core::CamRadtan>(320,240);Eigen::Matrix<double,8,1> k;
    k<<220,220,160,120,0,0,0,0;c->set_value(k);cameras[i]=c;
  }
  ov_core::TrackKLT tracker(cameras,120,0,true,ov_core::TrackBase::HISTOGRAM,10,4,3,8,3,15);
  tracker.set_gyro_mode("seed");tracker.set_flow_backend("cuda");const auto base=texture();
  Eigen::VectorXd left_pose(7);left_pose<<0,0,0,1,0,0,0;auto right_pose=left_pose;right_pose(4)=-.1;
  tracker.set_stereo_consistency("refine",{{0,left_pose},{1,right_pose}});
  for(int i=0;i<8;++i) {
    ov_core::CameraData data;data.timestamp=1+.05*i;data.sensor_ids={0,1};
    data.images={translated(base,.3*i,0),translated(base,-4+.3*i,0)};
    data.masks={cv::Mat::zeros(base.size(),CV_8UC1),cv::Mat::zeros(base.size(),CV_8UC1)};
    const Eigen::Matrix3d R=Eigen::Matrix3d::Identity();tracker.set_gyro_predictions(data.timestamp,{{0,R},{1,R}});
    tracker.feed_new_camera(data);
  }
  EXPECT_GT(tracker.get_last_ids().at(0).size(),40u);EXPECT_GT(tracker.get_last_ids().at(1).size(),40u);
  EXPECT_GT(tracker.stereo_statistics().refined,100u);
}
#endif
