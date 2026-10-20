cmake_minimum_required(VERSION 3.10)
project(ov_core)

# Include libraries (if we don't have opencv 4, then fallback to opencv 3)
# The OpenCV version needs to match the one used by cv_bridge otherwise you will get a segmentation fault!
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 QUIET)
if (NOT OpenCV_FOUND)
    find_package(OpenCV 4 REQUIRED)
endif ()
find_package(Boost REQUIRED CONFIG COMPONENTS filesystem thread date_time)
message(STATUS "OPENCV: " ${OpenCV_VERSION} " | BOOST: " ${Boost_VERSION})

# By default we build with ROS, but you can disable this and just build as a library
option(ENABLE_ROS "Enable or disable building with ROS (if it is found)" ON)

# If we will compile with aruco support
option(ENABLE_KLTNET_ONNX "Build the optional sparse learned tracker ONNX adapter (trained graph required)" OFF)
option(ENABLE_LETNET_ONNX "Build the optional author-trained LET-Net feature frontend" OFF)
option(ENABLE_LETNET_ONNXRUNTIME "Enable optional ONNX Runtime CPU inference for LET-Net" OFF)
option(ENABLE_CUDA_KLT "Build optional CUDA sparse KLT with an independent CUDA runtime component" OFF)
option(ENABLE_ARUCO_TAGS "Enable or disable aruco tag (disable if no contrib modules)" ON)
if (NOT ENABLE_ARUCO_TAGS)
    add_definitions(-DENABLE_ARUCO_TAGS=0)
    message(WARNING "DISABLING ARUCOTAG TRACKING!")
else ()
    add_definitions(-DENABLE_ARUCO_TAGS=1)
endif ()

# Ceres 2.2 requires C++17; ROS dependencies may request a newer standard.
if (NOT CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 17)
    set(CMAKE_CXX_STANDARD 17)
endif ()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to an optimized build; a requested build type (e.g. Debug) keeps its own optimization and debug flags
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif ()
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fno-signed-zeros;-fno-math-errno;-funroll-loops;-Wall;-Wuninitialized;-fno-omit-frame-pointer>")

# Find our ROS version!
# NOTE: Default to using the ROS1 package if both are in our enviroment
# NOTE: https://github.com/romainreignier/share_ros1_ros2_lib_demo
find_package(catkin QUIET COMPONENTS roscpp)
find_package(ament_cmake QUIET)
if (catkin_FOUND AND ENABLE_ROS)
    message(STATUS "ROS *1* version found, building ROS1.cmake")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS1.cmake)
elseif (ament_cmake_FOUND AND ENABLE_ROS)
    message(STATUS "ROS *2* version found, building ROS2.cmake")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS2.cmake)
else ()
    message(STATUS "No ROS versions found or building with ROS disabled, building ROS1.cmake without ROS")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS1.cmake)
endif ()





target_compile_definitions(ov_core_lib PUBLIC ENABLE_KLTNET_ONNX=$<BOOL:${ENABLE_KLTNET_ONNX}> ENABLE_LETNET_ONNX=$<BOOL:${ENABLE_LETNET_ONNX}>
    ENABLE_LETNET_ONNXRUNTIME=$<BOOL:${ENABLE_LETNET_ONNXRUNTIME}> ENABLE_CUDA_KLT=$<BOOL:${ENABLE_CUDA_KLT}>)
if(ENABLE_CUDA_KLT)
    if(CMAKE_VERSION VERSION_LESS 3.18)
        message(FATAL_ERROR "ENABLE_CUDA_KLT requires CMake 3.18 or newer")
    endif()
    cmake_policy(SET CMP0074 NEW)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(ov_klt_cuda SHARED src/track/cuda/CudaKLT.cu)
    set_target_properties(ov_klt_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
        INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")
    target_link_libraries(ov_klt_cuda PRIVATE CUDA::cudart)
    target_link_libraries(ov_core_lib PRIVATE ov_klt_cuda)
    set_property(TARGET ov_core_lib APPEND PROPERTY INSTALL_RPATH "$ORIGIN")
    install(TARGETS ov_klt_cuda EXPORT export_ov_core LIBRARY DESTINATION lib)
endif()
if(ENABLE_KLTNET_ONNX OR ENABLE_LETNET_ONNX)
    if(NOT TARGET opencv_dnn)
        message(FATAL_ERROR "Learned tracking requires the OpenCV dnn module")
    endif()
    target_link_libraries(ov_core_lib PRIVATE opencv_dnn)
endif()
if(ENABLE_LETNET_ONNXRUNTIME)
    set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime CPU SDK prefix")
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h HINTS "${ONNXRUNTIME_ROOT}" PATH_SUFFIXES include include/onnxruntime/core/session)
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime HINTS "${ONNXRUNTIME_ROOT}" PATH_SUFFIXES lib lib64)
    if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR "ENABLE_LETNET_ONNXRUNTIME requires the ONNX Runtime C++ SDK; set ONNXRUNTIME_ROOT")
    endif()
    target_include_directories(ov_core_lib PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
    target_link_libraries(ov_core_lib PRIVATE "${ONNXRUNTIME_LIBRARY}")
    get_filename_component(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
    set_property(TARGET ov_core_lib APPEND PROPERTY INSTALL_RPATH "${ONNXRUNTIME_LIBRARY_DIR}")
endif()
