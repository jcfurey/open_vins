# OpenVINS

[![ROS 1 Workflow](https://github.com/rpng/open_vins/actions/workflows/build_ros1.yml/badge.svg)](https://github.com/rpng/open_vins/actions/workflows/build_ros1.yml)
[![ROS 2 Workflow](https://github.com/rpng/open_vins/actions/workflows/build_ros2.yml/badge.svg)](https://github.com/rpng/open_vins/actions/workflows/build_ros2.yml)
[![ROS Free Workflow](https://github.com/rpng/open_vins/actions/workflows/build.yml/badge.svg)](https://github.com/rpng/open_vins/actions/workflows/build.yml)

Welcome to the OpenVINS project!
The OpenVINS project houses some core computer vision code along with a state-of-the art filter-based visual-inertial
estimator. The core filter is an [Extended Kalman filter](https://en.wikipedia.org/wiki/Extended_Kalman_filter) which
fuses inertial information with sparse visual feature tracks. These visual feature tracks are fused leveraging
the [Multi-State Constraint Kalman Filter (MSCKF)](https://ieeexplore.ieee.org/document/4209642) sliding window
formulation which allows for 3D features to update the state estimate without directly estimating the feature states in
the filter. Inspired by graph-based optimization systems, the included filter has modularity allowing for convenient
covariance management with a proper type-based state system. Please take a look at the feature list below for full
details on what the system supports.

* Github project page - https://github.com/rpng/open_vins
* Documentation - https://docs.openvins.com/
* Getting started guide - https://docs.openvins.com/getting-started.html
* Publication reference - https://pgeneva.com/downloads/papers/Geneva2020ICRA.pdf

## ROS 2 Lyrical host build

For Ubuntu 26.04 with ROS 2 Lyrical, run these commands from the workspace root
containing `src/open_vins`:

```bash
source /opt/ros/lyrical/setup.bash
rosdep install --from-paths src/open_vins --ignore-src --rosdistro lyrical -y
MAKEFLAGS='-j4' CMAKE_BUILD_PARALLEL_LEVEL=4 colcon build \
  --base-paths src/open_vins --symlink-install --executor sequential \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

This builds all five packages: `ov_core`, `ov_data`, `ov_init`, `ov_msckf`, and
`ov_eval`. The ROS 2 libraries export CMake targets and inherit the C++ standard
required by the installed ROS distribution. Boost.System is header-only and
does not require a separate library.

Start the stereo estimator with the bundled EuRoC calibration:

```bash
ros2 launch ov_msckf subscribe.launch.py config:=euroc_mav rviz_enable:=false
```

The estimator waits for images and IMU messages on the topics specified in the
configuration. Supply a calibrated configuration for your own sensors using
`config_path:=/absolute/path/to/estimator_config.yaml`.

Run the bundled simulation without a bag or camera:

```bash
ros2 run ov_msckf run_simulation \
  "$PWD/src/open_vins/config/euroc_mav/estimator_config.yaml" \
  --ros-args \
  --params-file "$PWD/src/open_vins/ov_msckf/launch/simulation_ros2.yaml" \
  -p sim_traj_path:="$PWD/src/open_vins/ov_data/sim/udel_gore.txt"
```

The parameter file supplies the simulator's required seeds, rates, and feature
generation settings.

### ROS 2 execution and sensor contracts

The live ROS 2 node uses two executor threads for sensor callbacks and one owned
estimator worker. Callbacks validate and queue measurements; initialization,
propagation, updates, and visualization all run on that worker. Shutdown joins
the worker before releasing estimator state or image transport publishers.
The legacy `multi_threading_subs` and `multi_threading_pubs` settings are forced
off in this pipeline. OpenCV can still parallelize tracking internally.
Fast odometry publishes at most once per consumed IMU batch; its configured
rate is a ceiling, and image processing can reduce the output rate. Every
accepted IMU sample still feeds the estimator.
When embedding the visualizer, stop and join the ROS executor before destroying
it so message-filter input callbacks have finished; then stop the estimator
worker. The installed executable performs this shutdown sequence.

IMU and image subscriptions use best-effort, volatile `SensorDataQoS`, which
also connects to reliable publishers. Camera dimensions must match the supplied
calibration. Stereo pairs must fall within `stereo_sync_tolerance`; independent
camera streams are ordered in the ingress queue. Frames wait until IMU samples
cover their timestamp plus the camera-to-IMU offset, and the worker receives
only the IMU up to the sample that covers the next frame, so processing order
matches an offline replay however far behind the worker runs. Propagation
requires full IMU interval coverage and never extrapolates missing endpoints.

The input queues and published paths are bounded. Camera overload drops the
oldest queued frame. IMU queue overflow resets the estimator because continuing
after silently losing inertial samples would invalidate its prediction. A
backward IMU timestamp jump larger than one second also starts a new estimator
epoch when `reset_on_time_jump` is enabled. Smaller stale or duplicate samples
are rejected. An initialized estimator resets after the configured IMU history
limit without a camera update; pre-initialization IMU history uses the
initialization window. Resets clear published path history. Shutdown logs ingress
counts, drops, invalid input, queue peaks, and ingress-triggered resets.
Reset calibration snapshots own independent camera models, so online intrinsic
updates cannot change the configured calibration used by later epochs.

The bundled EuRoC configuration uses static initialization with a stationary
window followed by motion. `init_dyn_use:=true` additionally permits dynamic
initialization when visual and inertial motion provide enough information;
the existing rotation, conditioning, and covariance checks remain active.
`try_zupt` also changes startup policy by allowing a stationary start without
waiting for motion, so it should be evaluated with the zero-velocity detector
and trajectory accuracy. These settings are not interchangeable.

`num_pts` is a total tracking target divided uniformly across the configured
cameras, rounded down per camera. For stereo, `num_pts:=200` requests a target
of 100 tracks per camera. `init_max_features` is split in the same way during
initialization. Optional loop/depth image drawing is skipped in ROS 2 without
loop/depth consumers; active geometry continues to update for later subscribers.

Loaded YAML settings and ROS overrides are declared as read-only startup
parameters. They can be inspected using ROS parameter services; changing them
requires restarting the node. `use_sim_time` retains its normal ROS behavior.
The following settings supplement the estimator configuration:

| Parameter | Default | Purpose |
| --- | --- | --- |
| `executor_threads` | `2` | Callback executor threads, from 1 to 16 |
| `max_imu_queue_size` | `4096` | Maximum pending inertial samples |
| `max_camera_queue_size` | `20` | Maximum pending camera packets; a stereo pair is one packet |
| `imu_qos_depth` / `camera_qos_depth` | `200` / `10` | DDS subscription history depths |
| `stereo_sync_tolerance` | `0.005` | Maximum stereo timestamp difference, seconds |
| `path_history_size` | `2000` | Maximum poses per published path; zero disables history |
| `odometry_publish_rate` | `200.0` | Maximum fast odometry/TF rate, Hz; zero disables it |
| `track_image_publish_rate` | `10.0` | Maximum diagnostic tracking image rate, Hz; zero disables it |
| `max_imu_history_seconds` | `10.0` | Initialized estimator history limit without camera updates |
| `reset_on_time_jump` | `true` | Reset on large backward IMU timestamp jumps |
| `topic_imu`, `topic_camera0`, `topic_camera1`, ... | Calibration topics | Explicit sensor topic aliases |
| `global_frame_id` / `imu_frame_id` | `global` / `imu` | Pose parent and odometry child frames |
| `camera_frame_prefix` / `truth_frame_id` | `cam` / `truth` | Camera TF prefix and ground-truth child frame |

Pose, odometry, TF, image, and feature headers use measurement time. Tracking
images retain camera acquisition time, including before initialization; state
outputs use the IMU clock. Pose
orientation covariance is expressed about fixed axes in the parent frame;
odometry twist is expressed in its child frame, with covariance including
orientation, velocity, and bias uncertainty. Path and cloud construction is
skipped without subscribers, and tracking visualization stays on the estimator
worker. The existing `loop_feats` interface retains its legacy `PointCloud`
message type for downstream compatibility; ordinary feature clouds use
`PointCloud2`.

### Regression checks

Build with `-DBUILD_TESTING=ON`, then run from the workspace root:

```bash
colcon test --base-paths src/open_vins --packages-select ov_core ov_init ov_msckf
colcon test-result --test-result-base build/ov_core --verbose
colcon test-result --test-result-base build/ov_init --verbose
colcon test-result --test-result-base build/ov_msckf --verbose
ROS_DOMAIN_ID=126 python3 src/open_vins/ov_msckf/test/ros2_runtime_smoke.py \
  --executable install/ov_msckf/lib/ov_msckf/run_subscribe_msckf \
  --config src/open_vins/config/euroc_mav/estimator_config.yaml \
  --output .colcon/open-vins-runtime-smoke
```

The C++ tests cover stereo exclusion masks and bearing accumulation, static
startup policy and bias recovery, history trimming, late image consumers,
independent reset calibration snapshots, IMU interval interpolation, missing
coverage, nonzero clock offsets, covariance frame conversion, invalid EKF
innovations, empty and populated PointCloud2 serialization, queue limits,
ordering, concurrent producers/consumers, and shutdown.
The runtime check exercises best-effort sensor publishers, topic aliases,
read-only parameters, camera overload, invalid IMU input, bag rewinds, and SIGINT.
`ov_msckf/test/ros2_euroc_output_smoke.py` additionally observes a live EuRoC
replay; its module docstring lists the required frame and path overrides. It
checks timestamps, frame IDs, finite covariances, path bounds, tracking image
rate, depth/color image storage, and CameraInfo calibration fields.

For memory checks, build a separate instrumented overlay and run the runtime
test with leak checking enabled:

```bash
colcon build --base-paths src/open_vins --packages-up-to ov_msckf \
  --build-base .colcon/open-vins-asan/build \
  --install-base .colcon/open-vins-asan/install --executor sequential \
  --cmake-args -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON \
  '-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer' \
  '-DCMAKE_CXX_FLAGS_DEBUG=-O1 -g1 -fno-omit-frame-pointer'
source .colcon/open-vins-asan/install/setup.bash
ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  colcon test --base-paths src/open_vins --packages-select ov_core ov_init ov_msckf \
    --build-base .colcon/open-vins-asan/build \
    --install-base .colcon/open-vins-asan/install
ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ROS_DOMAIN_ID=126 \
  python3 src/open_vins/ov_msckf/test/ros2_runtime_smoke.py \
    --executable .colcon/open-vins-asan/install/ov_msckf/lib/ov_msckf/run_subscribe_msckf \
    --config src/open_vins/config/euroc_mav/estimator_config.yaml \
    --output .colcon/open-vins-asan/runtime --raw-only
```

Use the matching `--build-base` and `--install-base` when testing an instrumented
overlay; selecting native libraries for an instrumented test can mix incompatible
Eigen allocation conventions.

On the verified Lyrical host, loading all installed image transports leaves a
157-byte ROS type-support allocation at exit, also reproduced with a standalone
`image_transport` publisher without OpenVINS. `--raw-only` isolates strict leak
checks from that external allocation; it does not disable leak detection.
Production transport selection remains configurable through standard ROS
parameters such as `trackhist.enable_pub_plugins:=[image_transport/raw]`, with
equivalent settings for `loop_depth` and `loop_depth_colored`.

Verified on 2026-10-16 with Ubuntu 26.04.1, ROS 2 Lyrical, CMake 4.2.3,
GCC 15.2, Boost 1.90, Ceres 2.2, and OpenCV 4.10. All five packages build,
and a separate CMake consumer links against the installed libraries. The stereo
launch accepts test images and IMU samples and exits cleanly on SIGINT. The
bundled UDel Gore simulation completes with 1,697 finite saved states over
169.6 seconds of trajectory, with position RMSE 0.068 m and orientation RMSE
0.201 degrees.

The ROS 2 hardening and initialization passes include 32 C++ regression tests, the ROS
runtime/parameter/shutdown checks, and 11 standalone queue tests under
ThreadSanitizer. ASan/UBSan/LeakSanitizer pass the C++ suite and a sensor replay
with raw image transports. Instrumented playback exercises bounded camera
overload, rather than serving as a throughput benchmark.

A native headless replay of the first 90 seconds of EuRoC `MH_01_easy` at 2x
accepts 17,830 IMU samples and 1,783 stereo packets without ingress drops or
resets. It saves 886 finite states spanning 44.25 seconds after initialization,
with position RMSE 0.073 m against interpolated ground truth after rigid SE(3)
alignment without scale. Peak RSS is about 116 MiB, compared with 323 MiB in the
original replay. The original run reported heap corruption at shutdown, so its
trajectory is not a trustworthy accuracy baseline. Normal-speed playback with
output subscribers also verifies acquisition timestamps, frame IDs, bounded
paths, depth image layout, and CameraInfo fields without ingress drops.

## News / Events

* **May 11, 2023** - Inertial intrinsic support released as part of v2.7 along with a few bug fixes and improvements to stereo KLT tracking. Please check out the [release page](https://github.com/rpng/open_vins/releases/tag/v2.7) for details.
* **April 15, 2023** - Minor update to v2.6.3 to support incremental feature triangulation of active features for downstream applications, faster zero-velocity update, small bug fixes, some example realsense configurations, and cached fast state prediction. Please check out the [release page](https://github.com/rpng/open_vins/releases/tag/v2.6.3) for details.
* **April 3, 2023** - We have released a monocular plane-aided VINS, termed [ov_plane](https://github.com/rpng/ov_plane), which leverages the OpenVINS project. Both now support the released [Indoor AR Table](https://github.com/rpng/ar_table_dataset) dataset.
* **July 14, 2022** - Improved feature extraction logic for >100hz tracking, some bug fixes and updated scripts. See v2.6.1 [PR#259](https://github.com/rpng/open_vins/pull/259) and v2.6.2 [PR#264](https://github.com/rpng/open_vins/pull/264).
* **March 14, 2022** - Initial dynamic initialization open sourcing, asynchronous subscription to inertial readings and publishing of odometry, support for lower frequency feature tracking. See v2.6 [PR#232](https://github.com/rpng/open_vins/pull/232) for details.
* **December 13, 2021** - New YAML configuration system, ROS2 support, Docker images, robust static initialization based on disparity, internal logging system to reduce verbosity, image transport publishers, dynamic number of features support, and other small fixes. See
  v2.5 [PR#209](https://github.com/rpng/open_vins/pull/209) for details.
* **July 19, 2021** - Camera classes, masking support, alignment utility, and other small fixes. See
  v2.4 [PR#117](https://github.com/rpng/open_vins/pull/186) for details.
* **December 1, 2020** - Released improved memory management, active feature pointcloud publishing, limiting number of
  features in update to bound compute, and other small fixes. See
  v2.3 [PR#117](https://github.com/rpng/open_vins/pull/117) for details.
* **November 18, 2020** - Released groundtruth generation utility package, [vicon2gt](https://github.com/rpng/vicon2gt)
  to enable creation of groundtruth trajectories in a motion capture room for evaulating VIO methods.
* **July 7, 2020** - Released zero velocity update for vehicle applications and direct initialization when standing
  still. See [PR#79](https://github.com/rpng/open_vins/pull/79) for details.
* **May 18, 2020** - Released secondary pose graph example
  repository [ov_secondary](https://github.com/rpng/ov_secondary) based
  on [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion). OpenVINS now publishes marginalized feature
  track, feature 3d position, and first camera intrinsics and extrinsics.
  See [PR#66](https://github.com/rpng/open_vins/pull/66) for details and discussion.
* **April 3, 2020** - Released [v2.0](https://github.com/rpng/open_vins/releases/tag/v2.0) update to the codebase with
  some key refactoring, ros-free building, improved dataset support, and single inverse depth feature representation.
  Please check out the [release page](https://github.com/rpng/open_vins/releases/tag/v2.0) for details.
* **January 21, 2020** - Our paper has been accepted for presentation in [ICRA 2020](https://www.icra2020.org/). We look
  forward to seeing everybody there! We have also added links to a few videos of the system running on different
  datasets.
* **October 23, 2019** - OpenVINS placed first in the [IROS 2019 FPV Drone Racing VIO Competition
  ](http://rpg.ifi.uzh.ch/uzh-fpv.html). We will be giving a short presentation at
  the [workshop](https://wp.nyu.edu/workshopiros2019mav/) at 12:45pm in Macau on November 8th.
* **October 1, 2019** - We will be presenting at the [Visual-Inertial Navigation: Challenges and Applications
  ](http://udel.edu/~ghuang/iros19-vins-workshop/index.html) workshop at [IROS 2019](https://www.iros2019.org/). The
  submitted workshop paper can be found at [this](http://udel.edu/~ghuang/iros19-vins-workshop/papers/06.pdf) link.
* **August 21, 2019** - Open sourced [ov_maplab](https://github.com/rpng/ov_maplab) for interfacing OpenVINS with
  the [maplab](https://github.com/ethz-asl/maplab) library.
* **August 15, 2019** - Initial release of OpenVINS repository and documentation website!

## Project Features

* Sliding window visual-inertial MSCKF
* Modular covariance type system
* Comprehensive documentation and derivations
* Extendable visual-inertial simulator
    * On manifold SE(3) b-spline
    * Arbitrary number of cameras
    * Arbitrary sensor rate
    * Automatic feature generation
* Five different feature representations
    1. Global XYZ
    2. Global inverse depth
    3. Anchored XYZ
    4. Anchored inverse depth
    5. Anchored MSCKF inverse depth
    6. Anchored single inverse depth
* Calibration of sensor intrinsics and extrinsics
    * Camera to IMU transform
    * Camera to IMU time offset
    * Camera intrinsics
    * Inertial intrinsics (including g-sensitivity)
* Environmental SLAM feature
    * OpenCV ARUCO tag SLAM features
    * Sparse feature SLAM features
* Visual tracking support
    * Monocular camera
    * Stereo camera (synchronized)
    * Binocular cameras (synchronized)
    * KLT or descriptor based
    * Masked tracking
* Static and dynamic state initialization
* Zero velocity detection and updates
* Out of the box evaluation on EuRocMav, TUM-VI, UZH-FPV, KAIST Urban and other VIO datasets
* Extensive evaluation suite (ATE, RPE, NEES, RMSE, etc..)

## Codebase Extensions

* **[ov_plane](https://github.com/rpng/ov_plane)** - A real-time monocular visual-inertial odometry (VIO) system which leverages
  environmental planes. At the core it presents an efficient robust monocular-based plane detection algorithm which does
  not require additional sensing modalities such as a stereo, depth camera or neural network. The plane detection and tracking
  algorithm enables real-time regularization of point features to environmental planes which are either maintained in the state
  vector as long-lived planes, or marginalized for efficiency. Planar regularities are applied to both in-state SLAM and
  out-of-state MSCKF point features, enabling long-term point-to-plane loop-closures due to the large spacial volume of planes.

* **[vicon2gt](https://github.com/rpng/vicon2gt)** - This utility was created to generate groundtruth trajectories using
  a motion capture system (e.g. Vicon or OptiTrack) for use in evaluating visual-inertial estimation systems.
  Specifically we calculate the inertial IMU state (full 15 dof) at camera frequency rate and generate a groundtruth
  trajectory similar to those provided by the EurocMav datasets. Performs fusion of inertial and motion capture
  information and estimates all unknown spacial-temporal calibrations between the two sensors.

* **[ov_maplab](https://github.com/rpng/ov_maplab)** - This codebase contains the interface wrapper for exporting
  visual-inertial runs from [OpenVINS](https://github.com/rpng/open_vins) into the ViMap structure taken
  by [maplab](https://github.com/ethz-asl/maplab). The state estimates and raw images are appended to the ViMap as
  OpenVINS runs through a dataset. After completion of the dataset, features are re-extract and triangulate with
  maplab's feature system. This can be used to merge multi-session maps, or to perform a batch optimization after first
  running the data through OpenVINS. Some example have been provided along with a helper script to export trajectories
  into the standard groundtruth format.

* **[ov_secondary](https://github.com/rpng/ov_secondary)** - This is an example secondary thread which provides loop
  closure in a loosely coupled manner for [OpenVINS](https://github.com/rpng/open_vins). This is a modification of the
  code originally developed by the HKUST aerial robotics group and can be found in
  their [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) repository. Here we stress that this is a
  loosely coupled method, thus no information is returned to the estimator to improve the underlying OpenVINS odometry.
  This codebase has been modified in a few key areas including: exposing more loop closure parameters, subscribing to
  camera intrinsics, simplifying configuration such that only topics need to be supplied, and some tweaks to the loop
  closure detection to improve frequency.


## Demo Videos

<a href="http://www.youtube.com/watch?v=KCX51GvYGss">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/KCX51GvYGss.jpg" width="120" height="90" />
</a>
<a href="http://www.youtube.com/watch?v=Lc7VQHngSuQ">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/Lc7VQHngSuQ.jpg" width="120" height="90" />
</a>
<a href="http://www.youtube.com/watch?v=vaia7iPaRW8">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/vaia7iPaRW8.jpg" width="120" height="90" />
</a>
<a href="http://www.youtube.com/watch?v=MCzTF9ye2zw">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/MCzTF9ye2zw.jpg"  width="120" height="90"/>
</a>
<a href="http://www.youtube.com/watch?v=eSQLWcNrx_I">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/eSQLWcNrx_I.jpg" width="120" height="90" />
</a>
<br/>

<a href="http://www.youtube.com/watch?v=187AXuuGNNw">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/187AXuuGNNw.jpg" width="120" height="90" />
</a>
<a href="http://www.youtube.com/watch?v=oUoLlrFryk0">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/oUoLlrFryk0.jpg" width="120" height="90" />
</a>
<a href="http://www.youtube.com/watch?v=ExPIGwORm4E">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/ExPIGwORm4E.jpg" width="120" height="90" />
</a>
<a href="http://www.youtube.com/watch?v=lXHl-qgLGl8">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/lXHl-qgLGl8.jpg" width="120" height="90" />
</a>



## Credit / Licensing

This code was written by the [Robot Perception and Navigation Group (RPNG)](https://sites.udel.edu/robot/) at the
University of Delaware. If you have any issues with the code please open an issue on our github page with relevant
implementation details and references. For researchers that have leveraged or compared to this work, please cite the
following:

```txt
@Conference{Geneva2020ICRA,
  Title      = {{OpenVINS}: A Research Platform for Visual-Inertial Estimation},
  Author     = {Patrick Geneva and Kevin Eckenhoff and Woosik Lee and Yulin Yang and Guoquan Huang},
  Booktitle  = {Proc. of the IEEE International Conference on Robotics and Automation},
  Year       = {2020},
  Address    = {Paris, France},
  Url        = {\url{https://github.com/rpng/open_vins}}
}
```

The codebase and documentation is licensed under the [GNU General Public License v3 (GPL-3)](https://www.gnu.org/licenses/gpl-3.0.txt).
You must preserve the copyright and license notices in your derivative work and make available the complete source code with modifications under the same license ([see this](https://choosealicense.com/licenses/gpl-3.0/); this is not legal advice).


