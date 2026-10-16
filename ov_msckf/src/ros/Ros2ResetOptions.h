/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef OV_MSCKF_ROS2_RESET_OPTIONS_H
#define OV_MSCKF_ROS2_RESET_OPTIONS_H

#include <stdexcept>
#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"
#include "core/VioManagerOptions.h"

namespace ov_msckf {

/// Take a calibration snapshot without sharing camera models mutated by online calibration.
inline VioManagerOptions copy_ros2_reset_options(const VioManagerOptions &source) {
  auto result = source;
  const auto clone_cameras = [](auto &cameras) {
    for (auto &[id, camera] : cameras) {
      std::shared_ptr<ov_core::CamBase> copy;
      if (std::dynamic_pointer_cast<ov_core::CamEqui>(camera))
        copy = std::make_shared<ov_core::CamEqui>(camera->w(), camera->h());
      else if (std::dynamic_pointer_cast<ov_core::CamRadtan>(camera))
        copy = std::make_shared<ov_core::CamRadtan>(camera->w(), camera->h());
      else
        throw std::invalid_argument("Unsupported camera model in ROS 2 reset calibration");
      copy->set_value(camera->get_value());
      camera = std::move(copy);
    }
  };
  clone_cameras(result.camera_intrinsics);
  clone_cameras(result.init_options.camera_intrinsics);
  return result;
}

} // namespace ov_msckf
#endif
