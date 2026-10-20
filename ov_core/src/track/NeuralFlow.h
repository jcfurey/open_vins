/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef OV_CORE_NEURAL_FLOW_H
#define OV_CORE_NEURAL_FLOW_H
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace ov_core {
/// Optional ONNX adapter for a complete sparse tracker, including KLTNet exports.
/// The graph contains coarse-flow prediction and triplet-patch refinement; this
/// adapter retains the fixed reference patch and previous displacement per track.
/// A trained graph with the documented interface is required, never synthesized or substituted.
class NeuralFlow {
public:
  struct Prediction {
    std::vector<cv::Point2f> points;
    std::vector<float> confidence;
  };
  explicit NeuralFlow(const std::string &model_path, const std::string &target = "cpu");
  ~NeuralFlow();
  Prediction predict(size_t camera, const cv::Mat &previous, const cv::Mat &current,
                     const std::vector<cv::Point2f> &points, const std::vector<size_t> &ids);
  void rekey(size_t camera, size_t old_id, size_t new_id);
  void retain(size_t camera, const std::vector<size_t> &ids);
  static bool available();

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace ov_core
#endif
