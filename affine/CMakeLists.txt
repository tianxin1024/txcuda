cmake_minimum_required(VERSION 2.6)

project(test_cuda)

# packages
find_package(CUDA QUIET REQUIRED)
find_package(OpenCV)

CUDA_ADD_LIBRARY(add_cuda warpAffine.cu)

add_executable(test_cuda warpAffine.cpp)

target_link_libraries(test_cuda add_cuda)

