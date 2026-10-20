/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "NeuralFlow.h"
#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <opencv2/imgproc.hpp>

#ifndef ENABLE_KLTNET_ONNX
#define ENABLE_KLTNET_ONNX 0
#endif
#if ENABLE_KLTNET_ONNX
#include <opencv2/dnn.hpp>
#endif

using namespace ov_core;
class NeuralFlow::Impl {
public:
  struct Track {
    cv::Mat reference;
    cv::Point2f previous_flow{0, 0};
  };
  std::unordered_map<size_t, std::unordered_map<size_t, Track>> tracks;
  std::mutex mutex;
#if ENABLE_KLTNET_ONNX
  cv::dnn::Net network;
#endif
};

bool NeuralFlow::available() { return ENABLE_KLTNET_ONNX; }

NeuralFlow::NeuralFlow(const std::string &model_path, const std::string &target) : impl(std::make_unique<Impl>()) {
#if ENABLE_KLTNET_ONNX
  if (model_path.empty())
    throw std::invalid_argument("kltnet_model_path must identify a trained ONNX export");
  impl->network = cv::dnn::readNetFromONNX(model_path);
  if (impl->network.empty())
    throw std::invalid_argument("The sparse-tracker ONNX graph is empty");
  if (target == "cpu") {
    impl->network.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    impl->network.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  } else if (target == "cuda") {
    const auto targets = cv::dnn::getAvailableTargets(cv::dnn::DNN_BACKEND_CUDA);
    if (std::find(targets.begin(), targets.end(), cv::dnn::DNN_TARGET_CUDA) == targets.end())
      throw std::invalid_argument("The requested OpenCV DNN CUDA backend is unavailable");
    impl->network.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    impl->network.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
  } else {
    throw std::invalid_argument("kltnet_target must be cpu or cuda");
  }
#else
  (void)model_path;
  (void)target;
  throw std::runtime_error("KLTNet ONNX support was not built; rebuild ov_core with -DENABLE_KLTNET_ONNX=ON and provide trained model assets");
#endif
}
NeuralFlow::~NeuralFlow() = default;

NeuralFlow::Prediction NeuralFlow::predict(size_t camera, const cv::Mat &previous, const cv::Mat &current,
                                          const std::vector<cv::Point2f> &points, const std::vector<size_t> &ids) {
#if ENABLE_KLTNET_ONNX
  if (points.size() != ids.size() || previous.type() != CV_8UC1 || current.type() != CV_8UC1 || previous.size() != current.size())
    throw std::invalid_argument("Invalid sparse ONNX tracker inputs");
  Prediction prediction;
  if (points.empty())
    return prediction;
  std::lock_guard<std::mutex> lock(impl->mutex); // cv::dnn::Net forward/setInput mutate shared inference storage.
  auto &cache = impl->tracks[camera];
  const int count = static_cast<int>(points.size());
  // Four axes avoid OpenCV DNN interpreting a three-axis tensor as a two-axis image.
  const int point_shape[]{1, 1, count, 2}, patch_shape[]{count, 1, 31, 31};
  cv::Mat point_tensor(4, point_shape, CV_32F), flow_tensor(4, point_shape, CV_32F), references(4, patch_shape, CV_32F);
  for (int i = 0; i < count; ++i) {
    if (!std::isfinite(points[i].x) || !std::isfinite(points[i].y))
      throw std::invalid_argument("Nonfinite input feature in sparse ONNX tracker");
    auto &track = cache[ids[i]];
    if (track.reference.empty()) {
      cv::getRectSubPix(previous, cv::Size(31, 31), points[i], track.reference, CV_32F);
      track.reference *= 1.0 / 255;
    }
    point_tensor.ptr<float>()[2 * i] = points[i].x;
    point_tensor.ptr<float>()[2 * i + 1] = points[i].y;
    flow_tensor.ptr<float>()[2 * i] = track.previous_flow.x;
    flow_tensor.ptr<float>()[2 * i + 1] = track.previous_flow.y;
    cv::Mat destination(31, 31, CV_32F, references.ptr<float>() + i * 31 * 31);
    track.reference.copyTo(destination);
  }
  impl->network.setInput(cv::dnn::blobFromImage(previous, 1.0 / 255, cv::Size(), cv::Scalar(), false, false), "image_previous");
  impl->network.setInput(cv::dnn::blobFromImage(current, 1.0 / 255, cv::Size(), cv::Scalar(), false, false), "image_current");
  impl->network.setInput(point_tensor, "points_previous");
  impl->network.setInput(flow_tensor, "flow_previous");
  impl->network.setInput(references, "patch_reference");
  std::vector<cv::Mat> outputs;
  impl->network.forward(outputs, std::vector<std::string>{"points_current", "confidence"});
  if (outputs.size() != 2 || outputs[0].type() != CV_32F || outputs[1].type() != CV_32F ||
      outputs[0].total() != points.size() * 2 || outputs[1].total() != points.size())
    throw std::runtime_error("Sparse ONNX tracker output shapes/types do not match the documented model contract");
  const cv::Mat output_points = outputs[0].isContinuous() ? outputs[0] : outputs[0].clone();
  const cv::Mat output_confidence = outputs[1].isContinuous() ? outputs[1] : outputs[1].clone();
  for (int i = 0; i < count; ++i) {
    const cv::Point2f point(output_points.ptr<float>()[2 * i], output_points.ptr<float>()[2 * i + 1]);
    prediction.points.push_back(point);
    prediction.confidence.push_back(output_confidence.ptr<float>()[i]);
    if (std::isfinite(point.x) && std::isfinite(point.y))
      cache[ids[i]].previous_flow = point - points[i];
  }
  return prediction;
#else
  (void)camera; (void)previous; (void)current; (void)points; (void)ids;
  throw std::runtime_error("Sparse ONNX inference support is disabled");
#endif
}

void NeuralFlow::rekey(size_t camera, size_t old_id, size_t new_id) {
  if (old_id == new_id)
    return;
  std::lock_guard<std::mutex> lock(impl->mutex);
  auto &cache = impl->tracks[camera];
  auto found = cache.find(old_id);
  if (found != cache.end()) {
    auto track = std::move(found->second);
    cache.erase(found);
    cache[new_id] = std::move(track);
  }
}

void NeuralFlow::retain(size_t camera, const std::vector<size_t> &ids) {
  std::lock_guard<std::mutex> lock(impl->mutex);
  const std::unordered_set<size_t> active(ids.begin(), ids.end());
  auto &cache = impl->tracks[camera];
  for (auto iterator = cache.begin(); iterator != cache.end();)
    if (!active.count(iterator->first))
      iterator = cache.erase(iterator);
    else
      ++iterator;
}
