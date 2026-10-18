cmake_minimum_required(VERSION 3.10)

# Find ros dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(ov_core REQUIRED)

# Describe ROS project
option(ENABLE_ROS "Enable or disable building with ROS (if it is found)" ON)
if (NOT ENABLE_ROS)
    message(FATAL_ERROR "Build with ROS1.cmake if you don't have ROS.")
endif ()
option(BUILD_OV_EXAMPLES "Build OpenVINS webcam, profiling and simulation examples" OFF)

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

##################################################
# Make the shared library
##################################################

list(APPEND LIBRARY_SOURCES
        src/dummy.cpp
        src/ceres/Factor_GenericPrior.cpp
        src/ceres/Factor_ImageReprojCalib.cpp
        src/ceres/Factor_ImuCPIv1.cpp
        src/ceres/State_JPLQuatLocal.cpp
        src/init/InertialInitializer.cpp
        src/dynamic/DynamicInitializer.cpp
        src/static/StaticInitializer.cpp
        src/sim/SimulatorInit.cpp
)
file(GLOB_RECURSE LIBRARY_HEADERS "src/*.h")
add_library(ov_init_lib SHARED ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
target_compile_definitions(ov_init_lib PUBLIC ROS_AVAILABLE=2)
target_link_libraries(ov_init_lib PUBLIC ov_core::ov_core_lib Ceres::ceres ${thirdparty_libraries})
target_include_directories(ov_init_lib PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
install(TARGETS ov_init_lib
        EXPORT export_ov_init
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include/${PROJECT_NAME}
)
install(DIRECTORY src/
        DESTINATION include/${PROJECT_NAME}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
ament_export_include_directories(include/${PROJECT_NAME})
ament_export_libraries(ov_init_lib)
ament_export_targets(export_ov_init HAS_LIBRARY_TARGET)
ament_export_dependencies(ov_core Ceres)

##################################################
# Make binary files!
##################################################

if(BUILD_OV_EXAMPLES)
    foreach(example test_simulation test_dynamic_mle test_dynamic_init)
        add_executable(${example} src/${example}.cpp)
        target_link_libraries(${example} ov_init_lib ${thirdparty_libraries})
        install(TARGETS ${example} DESTINATION lib/${PROJECT_NAME})
    endforeach()
endif()

# Install launch and config directories
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch/)

if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(test_initialization test/test_initialization.cpp)
    target_link_libraries(test_initialization ov_init_lib)
endif()

# finally define this as the package
ament_package()

