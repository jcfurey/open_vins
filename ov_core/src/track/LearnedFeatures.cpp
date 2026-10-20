/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "LearnedFeatures.h"
#include <algorithm>
#include <cmath>
#include <mutex>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#ifndef ENABLE_LETNET_ONNX
#define ENABLE_LETNET_ONNX 0
#endif
#if ENABLE_LETNET_ONNX
#include <opencv2/dnn.hpp>
#endif

using namespace ov_core;

class LearnedFeatures::Impl {
public:
  std::mutex mutex;
#if ENABLE_LETNET_ONNX
  cv::dnn::Net net;
#endif
};

bool LearnedFeatures::available() { return ENABLE_LETNET_ONNX; }
LearnedFeatures::~LearnedFeatures() = default;
LearnedFeatures::LearnedFeatures(const std::string &model_path) : impl(std::make_unique<Impl>()) {
#if ENABLE_LETNET_ONNX
  if (model_path.empty())
    throw std::invalid_argument("letnet_model_path must point to a prepared author-trained model");
  impl->net = cv::dnn::readNetFromONNX(model_path);
  if (impl->net.empty() || impl->net.getUnconnectedOutLayersNames() != std::vector<std::string>{"let_features"})
    throw std::invalid_argument("LET-Net requires the graph from prepare_open_vins_letnet.py (let_features output)");
  impl->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  impl->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
#else
  (void)model_path;
  throw std::runtime_error("LET-Net inference was not built; rebuild ov_core with -DENABLE_LETNET_ONNX=ON");
#endif
}

LearnedFeatures::Frame LearnedFeatures::infer(const cv::Mat &gray) {
#if ENABLE_LETNET_ONNX
  if (gray.empty() || gray.type() != CV_8UC1)
    throw std::invalid_argument("LET-Net expects a nonempty original-resolution grayscale CV_8UC1 image");
  std::lock_guard<std::mutex> lock(impl->mutex);
  impl->net.setInput(cv::dnn::blobFromImage(gray, 1.0 / 255));
  const cv::Mat blob = impl->net.forward("let_features");
  if (blob.type() != CV_32F || blob.dims != 4 || blob.size[0] != 1 || blob.size[1] != 4 || blob.size[2] != gray.rows ||
      blob.size[3] != gray.cols || !blob.isContinuous())
    throw std::runtime_error("LET-Net output must have shape [1,4,H,W] at the input image resolution");
  cv::Mat flat(1, static_cast<int>(blob.total()), CV_32F, const_cast<float *>(blob.ptr<float>()));
  if (!cv::checkRange(flat, true, nullptr, -1e-6, 1.000001))
    throw std::runtime_error("LET-Net produced nonfinite or non-sigmoid feature values");
  const size_t pixels = gray.total();
  std::vector<cv::Mat> channels;
  for (size_t channel = 0; channel < 3; ++channel)
    channels.emplace_back(gray.rows, gray.cols, CV_32F, const_cast<float *>(blob.ptr<float>() + channel * pixels));
  cv::Mat merged;
  cv::merge(channels, merged);
  Frame frame;
  merged.convertTo(frame.tracking, CV_8UC3, 255.0);
  // cv::dnn reuses its output workspace on the next camera; never retain a borrowed view.
  frame.score = cv::Mat(gray.rows, gray.cols, CV_32F, const_cast<float *>(blob.ptr<float>() + 3 * pixels)).clone();
  return frame;
#else
  (void)gray;
  throw std::runtime_error("LET-Net inference support is disabled");
#endif
}

void LearnedFeatures::detect(const cv::Mat &score, const cv::Mat &mask, const std::vector<std::pair<int, int>> &cells,
                             std::vector<cv::KeyPoint> &points, int per_cell, int grid_x, int grid_y, double minimum_score) {
  if (score.empty() || score.type() != CV_32FC1 || mask.type() != CV_8UC1 || score.size() != mask.size() || per_cell < 1 || grid_x < 1 ||
      grid_y < 1 || !std::isfinite(minimum_score) || minimum_score < 0 || minimum_score > 1)
    throw std::invalid_argument("Invalid learned-score grid or exclusion mask");
  cv::Mat maxima;
  cv::dilate(score, maxima, cv::Mat());
  constexpr int border = 10;
  for (const auto &cell : cells) {
    if (cell.first < 0 || cell.first >= grid_x || cell.second < 0 || cell.second >= grid_y)
      throw std::invalid_argument("Learned-score grid cell is outside the image grid");
    std::vector<cv::KeyPoint> candidates;
    const int begin_x = std::max(border, cell.first * score.cols / grid_x);
    const int end_x = std::min(score.cols - border, (cell.first + 1) * score.cols / grid_x);
    const int begin_y = std::max(border, cell.second * score.rows / grid_y);
    const int end_y = std::min(score.rows - border, (cell.second + 1) * score.rows / grid_y);
    for (int y = begin_y; y < end_y; ++y) {
      const float *values = score.ptr<float>(y), *peaks = maxima.ptr<float>(y);
      const uchar *excluded = mask.ptr<uchar>(y);
      for (int x = begin_x; x < end_x; ++x)
        if (excluded[x] <= 127 && std::isfinite(values[x]) && values[x] > 0 && values[x] >= minimum_score && values[x] == peaks[x])
          candidates.emplace_back(cv::Point2f(x, y), 1.0f, -1.0f, values[x]);
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) { return a.response > b.response; });
    const size_t count = std::min(candidates.size(), static_cast<size_t>(per_cell));
    points.insert(points.end(), candidates.begin(), candidates.begin() + count);
  }
}
