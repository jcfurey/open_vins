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

On Ubuntu 24.04 with ROS 2 Jazzy, use the same commands with `jazzy` in place
of `lyrical`. CI builds and tests Jazzy, Kilted, and Lyrical, plus a build
without ROS.

This builds all five packages: `ov_core`, `ov_data`, `ov_init`, `ov_msckf`, and
`ov_eval`. The ROS 2 libraries export CMake targets and inherit the C++ standard
required by the installed ROS distribution. Boost.System is header-only and
does not require a separate library.
Webcam, profiling and simulation tools are optional in ROS 2:
`BUILD_OV_EXAMPLES` defaults to `OFF`, independently of `BUILD_TESTING`.
`BUILD_OV_ROS2_REPLAY=OFF` also omits the rosbag2 serial tool and its CMake
dependency discovery; live and ASL file input remain available. Use a fresh
installation prefix when disabling targets to avoid stale installed binaries.
Rebuild C++ dependents together with the libraries when adopting updated public
tracker headers and layouts.

Start the stereo estimator with the bundled EuRoC calibration:

```bash
ros2 launch ov_msckf subscribe.launch.py config:=euroc_mav rviz_enable:=false
```

The estimator waits for images and IMU messages on the topics specified in the
configuration. Supply a calibrated configuration for your own sensors using
`config_path:=/absolute/path/to/estimator_config.yaml`.

EuRoC replay presets live under `config/euroc_mav/`. Select
`ros2_euroc_throughput_full.yaml` or `ros2_euroc_throughput_half.yaml` with the
`ros_params_file` launch argument for the bounded replay settings. The optional
`ros2_euroc_accuracy.yaml` uses denser tracking and KLT3/15 at higher CPU cost;
it preserves the same calibration, noise and reliable transport bounds.

`ros2_euroc_accuracy_dense.yaml` increases the budget to 600 tracked points,
25 clones, 60 SLAM landmarks and 120 MSCKF features per update. In one complete
11-flight EuRoC comparison at the original 20 Hz rate, mean position ATE changes
from 7.927 to 7.174 cm, CPU from 279.47 to 539.42 core-s, and median peak PSS
from 100.96 to 107.40 MiB. Worst-flight ATE increases from 16.083 to 16.980 cm.
Use this optional dataset-tuned preset when its extra computation is acceptable;
faster replay capacity is not qualified. It retains the original initialization,
calibration/noise policy and bounded transport. The 2 cm mean ATE target is not reached.

To run the bundled simulation without a bag or camera, first enable examples:

```bash
MAKEFLAGS='-j2 -l3' CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build \
  --base-paths src/open_vins --packages-select ov_core ov_init ov_msckf \
  --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_OV_EXAMPLES=ON
source install/setup.bash
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
Fast prediction and dynamic TF serialization run only when odometry or TF
consumers are present, including late subscribers on remapped topics. Fixed
camera extrinsics are retained on `/tf_static`; online extrinsic calibration
continues to publish dynamic transforms. Launch files default OpenMP/OpenBLAS
workers to one while honoring explicit environment settings.
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
| `klt_pyramid_levels` | `5` | Maximum optical-flow pyramid level, from 0 to 8 |
| `klt_window_size` | `15` | Odd optical-flow patch width, from 3 to 63 pixels |
| `klt_max_fb_error` | `0.0` | Experimental round-trip flow threshold in processed pixels; zero disables. Positive values also enable valid-only RANSAC inputs. |
| `klt_ransac_valid_only` | `false` | Experimental filtering of failed/invalid flow predictions before RANSAC. |
| `max_imu_history_seconds` | `10.0` | Initialized estimator history limit without camera updates |
| `reset_on_time_jump` | `true` | Reset on large backward IMU timestamp jumps |
| `topic_imu`, `topic_camera0`, `topic_camera1`, ... | Calibration topics | Explicit sensor topic aliases |
| `global_frame_id` / `imu_frame_id` | `global` / `imu` | Pose parent and odometry child frames |
| `camera_frame_prefix` / `truth_frame_id` | `cam` / `truth` | Camera TF prefix and ground-truth child frame |

The two experimental KLT checks remain off by default. A five-flight Machine Hall
accuracy screen found sequence regressions, so they are ablation tools rather than
a recommended accuracy preset. They preserve track indexing and do not use ground
truth. Synthetic tests cover invalid predictions, occlusion and round-trip tracking;
trajectory-level comparisons are still required for each proposed configuration.

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

### Optional research backends

`visual_update_backend` selects `standard` (default), `rdic`, `fej2`, or
`rdic_fej2`. The optional methods change the measurement model/update, and do
not consume ground truth. They are experimental and are not accuracy guarantees.
The matching EuRoC presets are `config/euroc_mav/ros2_euroc_rdic.yaml`,
`ros2_euroc_fej2.yaml`, and `ros2_euroc_rdic_fej2.yaml`; each retains the dense600
feature, clone, calibration, initialization, and transport settings.

* **RDIC-inspired uncertainty core:** robust per-frame tracking diagnostics
  (temporal photometric error, left/right imbalance, calibrated stereo epipolar
  error, bounded selective round-trip checks, and parallax change) set pixel
  covariance. Robust patch-gradient information adds directional uncertainty.
  Covariance is whitened **before** MSCKF landmark or SLAM bearing elimination,
  preserving correlated anisotropic measurement noise. A batch calibration uses
  the empirical normalized-innovation quantile collected before chi-square
  rejection to find the smallest bounded variance inflation. The implementation
  retains this fork's feature selection and budgets; it is not a full reproduction
  of the paper's feature-management policy. Shared state makes the block
  innovations correlated, so this empirical calibration is not a formal
  independent chi-square consistency test. Source:
  [RDIC-MSCKF (2026)](https://www.mdpi.com/2075-1702/14/9/1004).
* **FEJ2:** persistent SLAM measurements evaluate both current and first-estimate
  Jacobians, whiten the measurement noise, and project the update onto the left
  nullspace of the current-pose Jacobian discrepancy. QR determines its rank. Transient
  MSCKF features retain ordinary FEJ, as specified by the technical report's
  practical MSCKF-VIO design (§2.4.2). Anchor and calibration variables remain in
  the update; they are not additional discrepancy columns. A zero discrepancy
  preserves the update, while a full-row-rank discrepancy leaves no update.
  Projection follows the existing `max_slam_in_update` batches (25 in the EuRoC
  configuration); changing that size changes the information discarded per update.
  `use_fej: true` is required. Sources:
  [FEJ2 paper](https://pgeneva.com/downloads/papers/Chen2022ICRA.pdf),
  [technical report](https://yangyulin.net/papers/rpng_tr_2022_fej2.pdf).

| Parameter | Default | Meaning |
| --- | --- | --- |
| `visual_update_backend` | `standard` | Measurement update selection |
| `rdic_max_variance` | `4.0` | Maximum scalar **variance** inflation, minimum 1 |
| `rdic_quantile` | `0.8` | Empirical innovation calibration quantile |
| `rdic_min_blocks` | `6` | Minimum pre-gate feature blocks before calibration |
| `fej2_rank_tolerance` | `1e-6` | Relative rank threshold for the whitened discrepancy |
| `tracking_backend` | `native` | Existing KLT/descriptor selection, or `kltnet_onnx` |
| `kltnet_model_path` | empty | Required complete trained ONNX export |
| `kltnet_target` | `cpu` | OpenCV DNN `cpu` or available `cuda` target |
| `kltnet_min_confidence` | `0.1` | Export-specific relative-confidence rejection threshold |

#### Learned tracking integration

`tracking_backend: letnet_onnx` runs the author-trained grayscale
[LET-Net](https://github.com/linyicheng1/LET-NET) model. Learned corner scores
replace FAST detection and three learned feature channels replace grayscale
intensity for temporal and stereo LK. Raw grayscale images remain available for
visualization. All inference and tracking run in C++; no Python runtime is added
to the estimator. Camera resolution, pixel coordinates and calibration are retained.

Build the requested inference engine (`ENABLE_LETNET_ONNX` for OpenCV,
`ENABLE_LETNET_ONNXRUNTIME` for the CPU SDK) and rebuild all three VIO packages. Set
`histogram_method: NONE`, `letnet_model_path` to the prepared graph, and optionally
`letnet_min_score` (default 0.2, range [0,1]). The score cutoff is an empirical
corner-selection threshold, not a calibrated measurement uncertainty. The
`ros2_euroc_letnet_template.yaml` preset has an empty model path and requires a
prepared author model before startup. RDIC's photometric covariance currently
requires native grayscale LK, so LET-Net supports `standard` or `fej2` updates.

The workspace `scripts/prepare_open_vins_letnet.py --output <model.onnx>` downloads
the pinned author grayscale checkpoint and requires Python `onnx` for preparation
only. It checks the upstream SHA-256, preserves every trained initializer, and
exposes the existing sigmoid heads at dynamic image resolution. This removes
fixed-size Gather/Unsqueeze export handling without retraining, resizing camera
images or changing learned weights. A JSON receipt records input/output conventions
and both hashes. The upstream and prepared feature/score outputs are identical at
the original export resolution in the integration check. Weights stay outside the
source repository. The method is described in
[Breaking of brightness consistency in optical flow with a lightweight CNN network](https://arxiv.org/abs/2310.15655).

The workspace `scripts/benchmark_open_vins_learned.py --output <directory>` compares
histogram KLT, raw-image KLT and LET-Net at 1×, with equal thread counts and dense600
estimator budgets. It fingerprints the real model and all three loaded libraries,
preserves failures, and uses ground truth only for scoring. Runtime/accuracy must
be qualified on the target machine; the optional frontend is disabled by default.

For lower-overhead CPU inference, build with `-DENABLE_LETNET_ONNXRUNTIME=ON`
and `-DONNXRUNTIME_ROOT=<CPU SDK prefix>`, then select
`letnet_inference_backend: onnxruntime`. `letnet_inference_threads` defaults to
4 and bounds the runtime's private intra-operator pool independently of
`num_opencv_threads`; idle worker spinning is disabled. The same FP32 ONNX model
and trained weights are used, without quantization or image resizing. Real-model
tests compare its scores/features against OpenCV within floating-point precision
and verify stereo motion and output ownership. The SDK shared library must remain
available at the configured install path. Missing requested engines fail explicitly.

The original `letnet_inference_backend: opencv` remains selectable (built with
`ENABLE_LETNET_ONNX`). Its five full 1× Machine Hall flights exceeded the latency
gate and caused queue-overflow resets on this host, so the short pilot did not
qualify it for full-flight playback. The workspace report retains those failures.
Use `benchmark_open_vins_learned.py --engine onnxruntime --inference-threads 4`
with the matching install to validate the faster path. The validated CPU SDK for
this integration is the official ONNX Runtime 1.23.2 release.

The completed faster-engine screen kept up with all five full Machine Hall
flights at 1×: zero dropped inputs/resets and maximum P95 pose age 0.112 s.
Mean ATE was 12.62 cm for LET-Net versus 10.44 cm for fresh matched histogram
KLT controls. Results were mixed across sequences, so native remains the default.
The template selects the faster CPU engine and requires its SDK/build option.
The real-model comparison passes 104 VIO tests; the disabled core passes 24.
All failed OpenCV-DNN flights and a separately rerun paused-player startup are
retained in the workspace report. No repeatable accuracy gain is established.

The optional `kltnet_onnx` frontend loads a **complete trained sparse tracker**
through OpenCV DNN. It retains each feature's original reference patch and previous
displacement, handles feature-ID reassignment and dead-track cleanup, rejects
invalid/low-confidence predictions, and fits geometry only to valid predictions.
Stereo detection/matching remains native LK; temporal tracking uses the graph.
The learned confidence is used for rejection, not directly substituted for the
estimator's pixel covariance. This follows the OpenVINS integration approach in
[KLTNet (2026)](https://arxiv.org/html/2608.24544v1).

**This is an inference adapter, not an implementation or trained release of the
KLTNet neural architecture.** No author code/weights were available during this
integration. Running KLTNet requires trained assets exported/wrapped to this
explicit interface, including its coarse-flow network and triplet-patch refiner.
The included ONNX graph is a deterministic **test fixture only**, never a tracking
model or an accuracy baseline. Actual KLTNet accuracy and 1× throughput are untested.

Build `ov_core` with `-DENABLE_KLTNET_ONNX=ON` (default OFF), then rebuild `ov_init`
and `ov_msckf` against that same install. This work changes core feature metadata
and tracker interfaces, so rebuilding all three packages is required for any
backend. `ros2_euroc_kltnet_template.yaml` deliberately fails until a model path is
provided. Invalid/unavailable models or inference targets fail explicitly.

The graph contract uses float32 tensors, unrectified grayscale input with values
in `[0,1]`, pixel coordinates `(x,y)`, and dynamic track count `N`:

| Name | Shape | Meaning |
| --- | --- | --- |
| `image_previous`, `image_current` | `[1,1,H,W]` | Consecutive processed images |
| `points_previous` | `[1,1,N,2]` | Previous feature locations |
| `flow_previous` | `[1,1,N,2]` | Previous displacement; zero for a new track |
| `patch_reference` | `[N,1,31,31]` | Fixed initial patches; bilinear sampling, replicated borders |
| `points_current` (output) | `[1,1,N,2]` | Predicted absolute pixel locations |
| `confidence` (output) | `[N,1]` | Finite export-defined relative confidence |

Set histogram preprocessing and the confidence threshold to match the trained
model. The template uses `histogram_method: NONE`; image downsampling must also
match training. Export with operations supported by your OpenCV DNN build.

The workspace helper `scripts/benchmark_open_vins_backends.py --output <directory>`
runs all four visual backends on all five Machine Hall flights at 1×. It rotates
condition order, verifies actual core/initializer/estimator library hashes, and
retains input, continuity, coverage, error, and pose-age failures. Its default one
run per cell is exploratory; repeat runs before claiming an accuracy gain. The
RViz playlist accepts `--overlay <install> --ros-params-file <preset>` and retains
the ground-truth overlay and calibrated sensor model.

The 20 October 2026 Machine Hall screen completed all 20 full flights at 1× with
zero drops/resets and maximum P95 pose age below 0.093 s. Mean position ATE was
9.70 cm (`standard`), 12.90 cm (`rdic`), 11.20 cm (`fej2`), and 13.97 cm
(`rdic_fej2`). Each optional mode has sequence regressions; none was promoted.
This is one live run per cell, with ground truth used only by the scorer.
The enabled build passes 99 individual tests across core, initializer and
estimator; the ONNX-disabled core separately passes 22 tests. Full results and
source/library fingerprints are in the workspace report above.

### Optional gyro-assisted KLT

`gyro_klt_mode` is `none` by default. The independent experimental stages are
`seed`, `affine`, and `reference`, and require native KLT with standard observation
noise. They do not require a GPU or a neural model. All three keep the original
native initialization; assistance starts after the estimator is initialized.

`seed` integrates bias- and intrinsic-corrected gyro readings over the two camera
timestamps, including the current camera/IMU clock offset and camera extrinsic
rotation. Projection uses the camera's distortion model. This prediction does
not advance the state, covariance, clones, or fast-odometry cache. Missing IMU
coverage, internal gaps over 20 ms, and invalid intervals leave native LK active.

`affine` adds local patch-shape compensation and robust translation/gain/offset
refinement on the two finest **raw grayscale** pyramid levels. The ordinary
histogram-based pyramidal LK supplies coarse matching. The final measurement
remains a pixel in the original image, with translation determined visually.
`reference` additionally retains owned templates for up to
`gyro_klt_reference_age` frames (default 20). A failed old-template alignment
falls back to the independent frame-to-frame match and clears that reference.
Feature-ID reassignment, dead-track cleanup, image-buffer reuse, and unavailable
motion are handled explicitly. Cumulative runtime counters show which stages
actually ran. These are candidate accuracy changes, not established improvements.

Patch fits use a reusable bounded worker pool per camera, including the caller
as a participant. The participant count is `min(4, num_opencv_threads / cameras)`,
with a minimum of one. Independent fits can run concurrently inside OpenCV's
camera-parallel region; reference updates and feature-ID commits stay serial.
A regression test verifies identical serial/parallel pixels, errors and reference
counts. The initial serial reference screen exceeded the 0.5 s P95 pose-age gate
on three flights; those complete attempts are retained in the workspace study.
Use the separate optimized install for the final comparison.

The workspace helper `scripts/benchmark_open_vins_gyro.py --output <directory>`
runs all four modes twice on all five complete Machine Hall flights at 1× with
rotated condition order, equal dense600 budgets and verified loaded libraries.
Ground truth is consumed only by the scorer. The tests cover distorted point and
patch projection, subpixel motion with brightness changes, template ownership,
stereo feeds, bias/time-offset handling, and preservation of the filter state.

### Optional CUDA sparse KLT

`klt_flow_backend: cpu` remains the default. Build `ov_core` with
`-DENABLE_CUDA_KLT=ON`, then rebuild `ov_init` and `ov_msckf` against that same
install. Select `klt_flow_backend: cuda` and `klt_cuda_device: 0` at startup.
This independent native-grayscale backend can also accept gyro initial guesses.
Learned frontends are currently unsupported with CUDA KLT. Windows must be odd,
3..31 pixels, with pyramid levels 0..8. A requested unavailable backend fails
explicitly; it never silently selects CPU tracking.

The private `ov_klt_cuda` component uses the CUDA runtime through a small C ABI,
so a CUDA-enabled OpenCV build is unnecessary. Host images, camera models,
detection, geometric rejection and estimator updates use the existing ROS/OpenCV
libraries. Each camera owns its CUDA stream and reusable buffers. The GPU builds
Gaussian pyramids/Scharr derivatives and runs sparse inverse-compositional LK;
uploads, downloads and stream synchronization are included in timing. Numeric
results need not be identical to CPU OpenCV LK, so accuracy is evaluated separately.
`ENABLE_CUDA_KLT=OFF` builds require no CUDA toolkit or runtime.

The local RTX 3090 build uses CUDA 13.4, CMake 3.18 or newer and architecture 86:

```bash
colcon --log-base .colcon/open-vins-cuda-sdk/log build --base-paths src/open_vins \
  --packages-select ov_core ov_init ov_msckf --executor sequential \
  --build-base .colcon/open-vins-cuda-sdk/build --install-base .colcon/open-vins-cuda-sdk/install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DENABLE_CUDA_KLT=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.4 -DCMAKE_CUDA_ARCHITECTURES=86
```

Select a compatible toolkit/compiler and architecture for other hosts. The SDK
runtime must remain available at its configured path. GPU tests cover subpixel
motion, non-contiguous host images, resizing, failed/empty tracks, concurrent
camera streams, CPU comparison and stereo/gyro integration. NVIDIA memcheck and
racecheck logs are retained in the workspace study. The helper
`scripts/benchmark_open_vins_cuda.py --output <directory>` compares CPU/CUDA twice
on all five full Machine Hall flights, with gyro assistance disabled in both arms.
These acceleration results do not establish an accuracy gain.

The component binds its shared runtime to the compiler toolkit's implicit link
directories and records compiling/runtime versions at startup. Different runtime
major versions fail explicitly. The first local build used a cached system CUDA
12.4 runtime with the 13.4 compiler; that build and its initial flights are kept
as diagnostics. The primary trial uses a fresh matching-SDK prefix and verifies
the actual loaded runtime hash and major version for every CUDA flight.

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
reused KLT buffers versus fresh scratch storage through resets and configurable optical flow,
independent reset calibration snapshots, IMU interval interpolation, missing
coverage, nonzero clock offsets, covariance frame conversion, invalid EKF
innovations, empty and populated PointCloud2 serialization, queue limits,
ordering, concurrent producers/consumers, and shutdown.
The runtime check exercises best-effort sensor publishers, topic aliases,
read-only parameters, camera overload, invalid IMU input, bag rewinds, and SIGINT.
`ov_msckf/test/ros2_euroc_output_smoke.py` additionally observes a live EuRoC
replay; its module docstring lists the required frame and path overrides. It
checks timestamps, frame IDs, finite covariances, path bounds, tracking image
rate, depth/color image storage, CameraInfo calibration fields, and retained
static camera transforms for late listeners. Use its `--calibration-tf dynamic`
mode when testing online camera extrinsic calibration.
`ov_msckf/test/ros2_demand_smoke.py` replays a short EuRoC segment twice to check
zero fast predictions headless and returning TF/odometry for late consumers on
remapped topics; run it with `--install`, `--bag`, and a fresh `--output` path.
CI also configures `ov_msckf/test/installed_consumer` independently after
installation, builds against only the exported CMake target, and runs KLT and
enabled ArUco tracking to check transitive headers, feature flags and ABI.

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

### Deterministic replay and benchmarks

Live playback through DDS is timing dependent, so accuracy comparisons use the
serial runners. Both feed the estimator in timestamp order on one thread,
release each frame once IMU covers it (the live worker's rule), and never drop
input. A fixed input therefore always produces the same trajectory, and the
summary records a digest of every written value to prove it.

```bash
# ASL/EuRoC folders (EuRoC MAV, TUM-VI exports); also built without ROS
ros2 run ov_msckf asl_serial_msckf config/tum_vi/estimator_config.yaml \
  ~/datasets/tum_vi/dataset-room1_512_16 /tmp/room1 --verbosity WARNING

# ROS 2 bags (sqlite3 or mcap, raw or compressed images)
ros2 run ov_msckf ros2_serial_msckf config/tum_vi/estimator_config.yaml --ros-args \
  -p path_bag:=$HOME/bags/room1 -p output_dir:=/tmp/room1_bag
```

Each run writes `trajectory.txt` (ov_eval format with local orientation and
position covariances, IMU clock), `timing.csv`, and `summary.json` (counts,
per-frame timing, peak memory, digest). `ov_msckf/scripts/asl_to_ros2bag.py`
converts an ASL folder into a bag that replays to the identical digest.
`error_singlerun <align> <gt> <est> --json eval.json` writes ATE, RPE, NEES and
association coverage.

`ov_eval/benchmark/benchmark.py` wraps this into a reproducible protocol over
the sequences in `sequences.json` (TUM-VI rooms are downloadable; EuRoC must be
placed manually, its public host being rate limited):

```bash
python3 ov_eval/benchmark/benchmark.py fetch --data ~/datasets --only tum_vi_room1
python3 ov_eval/benchmark/benchmark.py run --data ~/datasets --results /tmp/bench --repeat 2
python3 ov_eval/benchmark/benchmark.py check --results /tmp/bench \
  --baseline ov_eval/benchmark/baselines/reference.json
```

`run` repeats every sequence and reports whether the trajectories are
bit-identical; `check` fails on ATE more than 10% (or 5 mm / 0.1 deg) above the
baseline, RPE more than 15% above it, lost coverage, or nondeterminism.
`--strict` also requires the baseline's digests, which only holds for the same
binaries and libraries. The `sim_udel_gore` entry needs no download: the
simulator writes its own ground truth, and CI gates on it.

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


