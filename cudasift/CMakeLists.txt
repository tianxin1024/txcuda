cmake_minimum_required(VERSION 2.6)

project(demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)
find_package(CUDA QUIET REQUIRED)

# CUDA_ADD_LIBRARY(demo_pro cudaSift.h cudaSiftH.cu cudautils.h cudaImage.h cudaImage.cu)
file(GLOB_RECURSE cuda_src ${PROJECT_SOURCE_DIR}/*.cu)
file(GLOB_RECURSE cuda_header ${PROJECT_SOURCE_DIR}/*.h)

cuda_add_library(demo_pro ${cuda_src} ${cuda_header})


add_executable(demo mainsift.cpp)

target_link_libraries(demo demo_pro)
target_link_libraries(demo ${OpenCV_LIBS})


