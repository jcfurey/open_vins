/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "CudaKLT.h"
#include <cmath>
#include <stdexcept>
#if ENABLE_CUDA_KLT
#include "cuda/CudaKltApi.h"
#endif
namespace ov_core {
struct CudaKLT::Impl {
#if ENABLE_CUDA_KLT
  OvCudaKlt *context=nullptr;
  std::vector<OvCudaPoint> previous,current;
  ~Impl() {ov_cuda_klt_destroy(context);}
#endif
};
CudaKLT::CudaKLT(int levels,int window,int device) : impl(new Impl) {
  if (levels<0 || levels>8 || window<3 || window>31 || window%2==0 || device<0)
    throw std::invalid_argument("CUDA KLT requires levels 0..8, odd window 3..31 and a nonnegative device");
#if ENABLE_CUDA_KLT
  char message[1024]={};impl->context=ov_cuda_klt_create(device,levels,window,message,sizeof(message));
  if (!impl->context) throw std::runtime_error(std::string("CUDA KLT startup failed: ")+message);
#else
  throw std::runtime_error("CUDA KLT was requested but this build has ENABLE_CUDA_KLT=OFF");
#endif
}
CudaKLT::~CudaKLT()=default;
int CudaKLT::runtime_version() {
#if ENABLE_CUDA_KLT
  return ov_cuda_klt_runtime_version();
#else
  return 0;
#endif
}
int CudaKLT::compiled_version() {
#if ENABLE_CUDA_KLT
  return ov_cuda_klt_compiled_version();
#else
  return 0;
#endif
}
bool CudaKLT::available() {
#if ENABLE_CUDA_KLT
  return ov_cuda_klt_device_count()>0;
#else
  return false;
#endif
}
std::string CudaKLT::device_name(int device) {
#if ENABLE_CUDA_KLT
  char name[256]={};if (ov_cuda_klt_device_name(device,name,sizeof(name))) return name;
#endif
  return {};
}
void CudaKLT::calc(const cv::Mat &previous,const cv::Mat &current,const std::vector<cv::Point2f> &points_previous,
                   std::vector<cv::Point2f> &points_current,std::vector<uchar> &status,std::vector<float> &error) {
  if (previous.empty() || previous.type()!=CV_8UC1 || current.type()!=CV_8UC1 || previous.size()!=current.size() ||
      points_current.size()!=points_previous.size())
    throw std::invalid_argument("CUDA KLT requires equal grayscale images and equally sized initial point vectors");
  status.assign(points_previous.size(),0);error.assign(points_previous.size(),0);
  if (points_previous.empty()) return;
#if ENABLE_CUDA_KLT
  impl->previous.resize(points_previous.size());impl->current.resize(points_current.size());
  for (size_t i=0;i<points_previous.size();++i) {
    impl->previous[i]={points_previous[i].x,points_previous[i].y};impl->current[i]={points_current[i].x,points_current[i].y};
  }
  char message[1024]={};
  if (!ov_cuda_klt_track(impl->context,previous.data,previous.step,current.data,current.step,previous.cols,previous.rows,
                          impl->previous.data(),impl->current.data(),status.data(),error.data(),points_previous.size(),message,sizeof(message)))
    throw std::runtime_error(std::string("CUDA KLT inference failed: ")+message);
  for (size_t i=0;i<points_current.size();++i) {
    const auto point=impl->current[i];
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(error[i])) status[i]=0;
    const bool old_finite=std::isfinite(points_previous[i].x) && std::isfinite(points_previous[i].y);
    points_current[i]=status[i] ? cv::Point2f(point.x,point.y) : (old_finite ? points_previous[i] : cv::Point2f());
  }
#else
  throw std::runtime_error("CUDA KLT is unavailable in this build");
#endif
}
} // namespace ov_core
