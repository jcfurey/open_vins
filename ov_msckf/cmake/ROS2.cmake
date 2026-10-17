cmake_minimum_required(VERSION 3.10)

# Find ROS build system
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(ov_core REQUIRED)
find_package(ov_init REQUIRED)

# Describe ROS project
option(ENABLE_ROS "Enable or disable building with ROS (if it is found)" ON)
if (NOT ENABLE_ROS)
    message(FATAL_ERROR "Build with ROS1.cmake if you don't have ROS.")
endif ()
add_definitions(-DROS_AVAILABLE=2)

# Include our header files
include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${CERES_LIBRARIES}
        ${OpenCV_LIBRARIES}
)
list(APPEND ros_libraries
        rclcpp::rclcpp
        ${rcl_interfaces_TARGETS}
        tf2::tf2
        tf2_ros::tf2_ros
        tf2_geometry_msgs::tf2_geometry_msgs
        ${std_msgs_TARGETS}
        ${geometry_msgs_TARGETS}
        ${sensor_msgs_TARGETS}
        ${nav_msgs_TARGETS}
        cv_bridge::cv_bridge
        image_transport::image_transport
        message_filters::message_filters
        ov_core::ov_core_lib
        ov_init::ov_init_lib
)

##################################################
# Make the shared library
##################################################

list(APPEND LIBRARY_SOURCES
        src/dummy.cpp
        src/sim/Simulator.cpp
        src/state/State.cpp
        src/state/StateHelper.cpp
        src/state/Propagator.cpp
        src/core/VioManager.cpp
        src/core/VioManagerHelper.cpp
        src/update/UpdaterHelper.cpp
        src/update/UpdaterMSCKF.cpp
        src/update/UpdaterSLAM.cpp
        src/update/UpdaterZeroVelocity.cpp
        src/replay/AslDataset.cpp
        src/replay/ReplaySummary.cpp
        src/replay/SerialFeeder.cpp
        src/replay/TrajectoryWriter.cpp
)
list(APPEND LIBRARY_SOURCES src/ros/ROS2Visualizer.cpp src/ros/ROSVisualizerHelper.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS "src/*.h")
add_library(ov_msckf_lib SHARED ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
target_compile_definitions(ov_msckf_lib PUBLIC ROS_AVAILABLE=2)
target_link_libraries(ov_msckf_lib PUBLIC ${ros_libraries} ${thirdparty_libraries})
target_include_directories(ov_msckf_lib PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)
install(TARGETS ov_msckf_lib
        EXPORT export_ov_msckf
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include
)
install(DIRECTORY src/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
ament_export_include_directories(include)
ament_export_libraries(ov_msckf_lib)
ament_export_targets(export_ov_msckf HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp rcl_interfaces tf2 tf2_ros tf2_geometry_msgs std_msgs geometry_msgs sensor_msgs nav_msgs cv_bridge image_transport message_filters ov_core ov_init Ceres)

##################################################
# Make binary files!
##################################################

add_executable(run_subscribe_msckf src/run_subscribe_msckf.cpp)
target_link_libraries(run_subscribe_msckf ov_msckf_lib ${thirdparty_libraries})
install(TARGETS run_subscribe_msckf DESTINATION lib/${PROJECT_NAME})

add_executable(asl_serial_msckf src/asl_serial_msckf.cpp)
target_link_libraries(asl_serial_msckf ov_msckf_lib ${thirdparty_libraries})
install(TARGETS asl_serial_msckf DESTINATION lib/${PROJECT_NAME})

add_executable(run_simulation src/run_simulation.cpp)
target_link_libraries(run_simulation ov_msckf_lib ${thirdparty_libraries})
install(TARGETS run_simulation DESTINATION lib/${PROJECT_NAME})

add_executable(test_sim_meas src/test_sim_meas.cpp)
target_link_libraries(test_sim_meas ov_msckf_lib ${thirdparty_libraries})
install(TARGETS test_sim_meas DESTINATION lib/${PROJECT_NAME})

add_executable(test_sim_repeat src/test_sim_repeat.cpp)
target_link_libraries(test_sim_repeat ov_msckf_lib ${thirdparty_libraries})
install(TARGETS test_sim_repeat DESTINATION lib/${PROJECT_NAME})

# Install launch and config directories
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch/)
install(DIRECTORY ../config/ DESTINATION share/${PROJECT_NAME}/config/)

if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(test_ros2_measurement_buffer test/test_ros2_measurement_buffer.cpp)
    target_link_libraries(test_ros2_measurement_buffer ov_msckf_lib)
    ament_add_gtest(test_estimation_regressions test/test_estimation_regressions.cpp)
    target_link_libraries(test_estimation_regressions ov_msckf_lib)
    ament_add_gtest(test_serial_replay test/test_serial_replay.cpp)
    target_link_libraries(test_serial_replay ov_msckf_lib)
endif()

# finally define this as the package
ament_package()
