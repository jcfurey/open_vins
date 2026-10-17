cmake_minimum_required(VERSION 3.10)

# Find ros dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)

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
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
)

##################################################
# Make the core library
##################################################

list(APPEND LIBRARY_SOURCES
        src/dummy.cpp
        src/cpi/CpiV1.cpp
        src/cpi/CpiV2.cpp
        src/sim/BsplineSE3.cpp
        src/track/TrackBase.cpp
        src/track/TrackAruco.cpp
        src/track/TrackDescriptor.cpp
        src/track/TrackKLT.cpp
        src/track/TrackSIM.cpp
        src/types/Landmark.cpp
        src/feat/Feature.cpp
        src/feat/FeatureDatabase.cpp
        src/feat/FeatureInitializer.cpp
        src/utils/print.cpp
)
file(GLOB_RECURSE LIBRARY_HEADERS "src/*.h")
add_library(ov_core_lib SHARED ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
target_compile_definitions(ov_core_lib PUBLIC ROS_AVAILABLE=2)
target_link_libraries(ov_core_lib PUBLIC rclcpp::rclcpp ${rcl_interfaces_TARGETS} cv_bridge::cv_bridge Eigen3::Eigen ${thirdparty_libraries})
target_include_directories(ov_core_lib PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
install(TARGETS ov_core_lib
        EXPORT export_ov_core
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include/${PROJECT_NAME}
)
install(DIRECTORY src/
        DESTINATION include/${PROJECT_NAME}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
ament_export_include_directories(include/${PROJECT_NAME})
ament_export_libraries(ov_core_lib)
ament_export_targets(export_ov_core HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 OpenCV rclcpp rcl_interfaces cv_bridge)

##################################################
# Make binary files!
##################################################

# TODO: UPGRADE THIS TO ROS2 AS ANOTHER FILE!!
#if (catkin_FOUND AND ENABLE_ROS)
#    add_executable(test_tracking src/test_tracking.cpp)
#    target_link_libraries(test_tracking ov_core_lib ${thirdparty_libraries})
#endif ()

add_executable(test_webcam src/test_webcam.cpp)
target_link_libraries(test_webcam ov_core_lib ${thirdparty_libraries})
install(TARGETS test_webcam DESTINATION lib/${PROJECT_NAME})

add_executable(test_profile src/test_profile.cpp)
target_link_libraries(test_profile ov_core_lib ${thirdparty_libraries})
install(TARGETS test_profile DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(test_stereo_mask test/test_stereo_mask.cpp)
    target_link_libraries(test_stereo_mask ov_core_lib)
    ament_add_gtest(test_binocular_tracking test/test_binocular_tracking.cpp)
    target_link_libraries(test_binocular_tracking ov_core_lib)
    ament_add_gtest(test_dataset_reader test/test_dataset_reader.cpp)
    target_link_libraries(test_dataset_reader ov_core_lib)
    ament_add_gtest(test_camera_models test/test_camera_models.cpp)
    target_link_libraries(test_camera_models ov_core_lib)
endif()

# finally define this as the package
ament_package(CONFIG_EXTRAS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Dependencies.cmake")
