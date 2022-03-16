cmake_minimum_required(VERSION 2.6)

project(demo)

find_package(OpenCV REQUIRED)
find_package(CUDA QUIET REQUIRED)

CUDA_ADD_LIBRARY(demo_pro cudaSift.cuh cudaSiftH.cu)

add_executable(demo mainsift.cpp)

target_link_libraries(demo demo_pro)
target_link_libraries(demo ${OpenCV_LIBS})


