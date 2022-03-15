cmake_minimum_required(VERSION 3.0)

project(test_cuda)

# packages
find_package(CUDA)

# nvcc flags
set(CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_20;-G;-g)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

set(CMAKE_NVCC_FLAGS "CMAKE_NVCC_FLAGS -std=c++11")

CUDA_ADD_EXECUTABLE(test_cuda ${CURRENT_HEADERS} ${CURRENT_SOURCES})

