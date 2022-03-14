cmake_minimum_required(VERSION 3.0)

project(cuda_pro)

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4/")
set(CUDA_DIR "/usr/local/cuda-10.2")

find_package(CUDA REQUIRED)
find_package(OpenCV)

add_executable(demo warpAffine.cpp)
target_link_libraries(demo ${OpenCV_LIBS})

