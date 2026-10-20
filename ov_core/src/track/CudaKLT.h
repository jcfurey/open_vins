/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef OV_CORE_CUDA_KLT_H
#define OV_CORE_CUDA_KLT_H
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
namespace ov_core {
/// Optional CUDA sparse LK. Its private C ABI keeps CUDA and OpenCV builds independent.
/// Each instance owns its stream and buffers; use one instance per concurrently tracked camera.
class CudaKLT {
public:
  CudaKLT(int levels, int window, int device = 0);
  ~CudaKLT();
  CudaKLT(const CudaKLT &) = delete;
  CudaKLT &operator=(const CudaKLT &) = delete;
  static bool available();
  static std::string device_name(int device = 0);
  static int runtime_version();
  static int compiled_version();
  void calc(const cv::Mat &previous, const cv::Mat &current, const std::vector<cv::Point2f> &points_previous,
            std::vector<cv::Point2f> &points_current, std::vector<uchar> &status, std::vector<float> &error);
private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace ov_core
#endif
