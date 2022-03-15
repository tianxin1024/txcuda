cmake_minimum_required(VERSION 2.6)

project(demo)

find_package(CUDA QUIET REQUIRED)

CUDA_ADD_LIBRARY(add_demo Math.h Math.cpp add.cuh add.cu)

add_executable(demo main.cpp)

target_link_libraries(demo add_demo)
