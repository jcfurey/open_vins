/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef OV_CORE_LEARNED_FEATURES_H
#define OV_CORE_LEARNED_FEATURES_H

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <utility>
#include <vector>

namespace ov_core {
/// LET-Net inference with author-trained weights. See scripts/prepare_open_vins_letnet.py
/// for the explicit dynamic-resolution graph contract and unchanged-weight receipt.
class LearnedFeatures {
public:
  struct Frame {
    cv::Mat tracking; ///< Three learned feature channels, CV_8UC3, at original pixel resolution.
    cv::Mat score;    ///< Learned corner scores, CV_32FC1, [0,1]. Owns its storage across inference calls.
  };
  explicit LearnedFeatures(const std::string &model_path, const std::string &engine = "opencv", int threads = 4);
  ~LearnedFeatures();
  Frame infer(const cv::Mat &gray);
  static bool available(const std::string &engine = "opencv");

  /// Grid-balanced local maxima with the fork's exclusion-mask convention (255 excludes).
  static void detect(const cv::Mat &score, const cv::Mat &mask, const std::vector<std::pair<int, int>> &cells,
                     std::vector<cv::KeyPoint> &points, int per_cell, int grid_x, int grid_y, double minimum_score);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace ov_core
#endif
