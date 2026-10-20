/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <gtest/gtest.h>
#include "cam/CamRadtan.h"
#include "cam/CamEqui.h"
#include "track/GyroKLT.h"
#include "track/TrackKLT.h"
#include "utils/quat_ops.h"

namespace {
std::shared_ptr<ov_core::CamBase> camera(bool fisheye=false) {
  std::shared_ptr<ov_core::CamBase> c;
  if (fisheye) c=std::make_shared<ov_core::CamEqui>(320,240);
  else c=std::make_shared<ov_core::CamRadtan>(320,240);
  Eigen::Matrix<double,8,1> k;
  k<<220,220,160,120,-.12,.02,.001,-.001;c->set_value(k);return c;
}
std::shared_ptr<ov_core::CamBase> pinhole() {
  auto c=camera();Eigen::Matrix<double,8,1> k;k<<220,220,160,120,0,0,0,0;c->set_value(k);return c;
}
std::vector<cv::Mat> pyramid(const cv::Mat &image) {
  std::vector<cv::Mat> result;cv::buildOpticalFlowPyramid(image,result,cv::Size(15,15),1);return result;
}
cv::Mat texture() {
  cv::Mat result(240,320,CV_8UC1);cv::RNG rng(123);rng.fill(result,cv::RNG::UNIFORM,30,200);
  cv::GaussianBlur(result,result,cv::Size(3,3),.6);return result;
}
cv::Mat transform(const cv::Mat &image,double angle,cv::Point2f translation,double gain=1,double bias=0) {
  const double c=std::cos(angle),s=std::sin(angle);
  cv::Mat affine=(cv::Mat_<double>(2,3)<<c,-s,160-c*160+s*120+translation.x,
                                     s,c,120-s*160-c*120+translation.y);
  cv::Mat result;cv::warpAffine(image,result,affine,image.size());result.convertTo(result,CV_8U,gain,bias);return result;
}
std::vector<cv::Point2f> points() {
  std::vector<cv::Point2f> result;
  for(int y=45;y<200;y+=25) for(int x=45;x<280;x+=25) result.emplace_back(x,y);
  return result;
}
std::vector<size_t> ids(size_t count) {std::vector<size_t> result;for(size_t i=0;i<count;++i)result.push_back(i+100);return result;}
}

TEST(GyroProjection, RadtanAndFisheyeUseDistortedCoordinatesAndCorrectRotationDirection) {
  for (const bool fisheye : {false,true}) {
    auto c=camera(fisheye);const auto R=ov_core::exp_so3(Eigen::Vector3d(.035,-.02,.04));
    const cv::Point2f point(275,175);cv::Point2f predicted;Eigen::Matrix2d affine;
    ASSERT_TRUE(ov_core::GyroKLT::project(*c,R,point,predicted,&affine));
    const auto norm=c->undistort_d(Eigen::Vector2d(point.x,point.y));
    const Eigen::Vector3d ray=R*Eigen::Vector3d(norm.x(),norm.y(),1);
    const auto expected=c->distort_d(ray.head<2>()/ray.z());
    EXPECT_NEAR(cv::norm(predicted-cv::Point2f(expected.x(),expected.y())),0,1e-4);
    const Eigen::Vector2d shift(.1,-.15);cv::Point2f shifted;
    ASSERT_TRUE(ov_core::GyroKLT::project(*c,R,point+cv::Point2f(shift.x(),shift.y()),shifted));
    EXPECT_LT((Eigen::Vector2d(shifted.x-predicted.x,shifted.y-predicted.y)-affine*shift).norm(),.002);
  }
}

TEST(GyroKlt, AffinePhotometricRefinementRecoversSubpixelMotionAndBrightnessChanges) {
  const auto before=pyramid(texture());const double angle=.035;
  const auto after=pyramid(transform(before[0],angle,cv::Point2f(.35,-.2),1.15,7));
  const auto previous=points();auto current=previous;const auto feature_ids=ids(previous.size());
  ov_core::GyroKLT tracker(pinhole(),"affine",15);
  const auto R=ov_core::exp_so3(Eigen::Vector3d(0,0,angle));tracker.begin_frame(&R);tracker.seed(previous,current);
  // Deliberately leave a translation error: the pixels, rather than the gyro, must resolve it.
  std::vector<uchar> status(previous.size(),1);std::vector<float> error(previous.size(),0);
  tracker.refine(before,after,previous,feature_ids,current,status,error);
  EXPECT_GT(tracker.statistics().refinements,previous.size()*3/4);
  double sum=0;
  for(size_t i=0;i<previous.size();++i) {
    cv::Point2f expected;ASSERT_TRUE(ov_core::GyroKLT::project(*pinhole(),R,previous[i],expected));
    sum+=cv::norm(current[i]-expected-cv::Point2f(.35,-.2));
  }
  EXPECT_LT(sum/previous.size(),.12);
}

TEST(GyroKlt, RetainedTemplatesOwnPixelsRekeyAndClearAcrossMissingMotion) {
  cv::Mat base=texture();auto before=pyramid(base);const double angle=.025;
  auto middle=pyramid(transform(base,angle,{}));auto after=pyramid(transform(base,2*angle,{}));
  const auto previous=points();auto first=previous;const auto feature_ids=ids(previous.size());
  ov_core::GyroKLT tracker(pinhole(),"reference",15);
  const auto R=ov_core::exp_so3(Eigen::Vector3d(0,0,angle));tracker.begin_frame(&R);tracker.seed(previous,first);
  std::vector<uchar> status(previous.size(),1);std::vector<float> error(previous.size(),0);
  tracker.refine(before,middle,previous,feature_ids,first,status,error);
  ASSERT_GT(tracker.statistics().refinements,previous.size()/2);
  before[0].setTo(0); // Simulate reuse of the previous image's backing storage.
  tracker.rekey(feature_ids[0],9999);auto next_ids=feature_ids;next_ids[0]=9999;
  auto second=first;tracker.begin_frame(&R);tracker.seed(first,second);
  tracker.refine(middle,after,first,next_ids,second,status,error);
  EXPECT_GT(tracker.statistics().references,previous.size()/2);
  const auto count=tracker.statistics().references;
  tracker.begin_frame(nullptr);EXPECT_FALSE(tracker.valid());
  tracker.begin_frame(&R);tracker.refine(middle,after,first,next_ids,second,status,error);
  EXPECT_EQ(tracker.statistics().references,count);
  tracker.retain({});
}

TEST(GyroKlt, InvalidRotationAndUnavailablePredictionPreserveNativeInitialGuess) {
  ov_core::GyroKLT tracker(pinhole(),"seed",15);auto old=points(),next=old;
  Eigen::Matrix3d bad=2*Eigen::Matrix3d::Identity();tracker.begin_frame(&bad);tracker.seed(old,next);
  EXPECT_FALSE(tracker.valid());EXPECT_EQ(old,next);
  cv::Point2f prediction;
  const auto behind=ov_core::exp_so3(Eigen::Vector3d(0,3.141592653589793,0));
  EXPECT_FALSE(ov_core::GyroKLT::project(*pinhole(),behind,cv::Point2f(160,120),prediction));
  EXPECT_THROW(ov_core::GyroKLT(pinhole(),"other",15),std::invalid_argument);
}

TEST(GyroKlt, StereoFeedsKeepRawViewsAndFeatureIdsAcrossReferenceTracking) {
  std::unordered_map<size_t,std::shared_ptr<ov_core::CamBase>> cams{{0,pinhole()},{1,pinhole()}};
  ov_core::TrackKLT tracker(cams,120,0,true,ov_core::TrackBase::HISTOGRAM,10,4,3,8,3,15);
  tracker.set_gyro_mode("reference");const cv::Mat base=texture();
  for(int i=0;i<8;++i) {
    ov_core::CameraData data;data.timestamp=1+.05*i;data.sensor_ids={0,1};
    data.images={transform(base,0,cv::Point2f(i*.25,0)),transform(base,0,cv::Point2f(-4+i*.25,0))};
    data.masks={cv::Mat::zeros(base.size(),CV_8UC1),cv::Mat::zeros(base.size(),CV_8UC1)};
    const Eigen::Matrix3d R=Eigen::Matrix3d::Identity();tracker.set_gyro_predictions(data.timestamp,{{0,R},{1,R}});
    tracker.feed_new_camera(data);
    EXPECT_DOUBLE_EQ(tracker.last_frame_time(0),data.timestamp);
  }
  EXPECT_GT(tracker.get_last_ids().at(0).size(),40u);
  EXPECT_GT(tracker.get_last_ids().at(1).size(),40u);
}
