cmake_minimum_required(VERSION 3.16)
project(open_vins_installed_consumer LANGUAGES CXX)

# Configure independently after installation: no source-tree include paths or flags.
find_package(ov_msckf REQUIRED)
add_executable(open_vins_installed_consumer client.cpp)
target_compile_features(open_vins_installed_consumer PRIVATE cxx_std_17)
target_link_libraries(open_vins_installed_consumer PRIVATE ov_msckf::ov_msckf_lib)
