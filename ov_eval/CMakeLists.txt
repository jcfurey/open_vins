cmake_minimum_required(VERSION 3.10)
project(ov_eval)

# Include libraries
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED CONFIG COMPONENTS filesystem thread date_time)

# By default we build with ROS, but you can disable this and just build as a library
option(ENABLE_ROS "Enable or disable building with ROS (if it is found)" ON)

# Plots embed Python through matplotlib-cpp (needs the Python 3 headers and library, and matplotlib at runtime)
option(DISABLE_MATPLOTLIB "Disable or enable matplotlib plot scripts in ov_eval" OFF)
if (NOT DISABLE_MATPLOTLIB)
    find_package(Python3 QUIET COMPONENTS Development)
endif ()
if (Python3_FOUND)
    add_definitions(-DHAVE_PYTHONLIBS=1)
    message(STATUS "PYTHON VERSION: " ${Python3_VERSION})
    include_directories(${Python3_INCLUDE_DIRS})
    list(APPEND thirdparty_libraries ${Python3_LIBRARIES})
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
add_compile_options(-fno-signed-zeros -fno-math-errno -funroll-loops -Wall -Wuninitialized)

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

