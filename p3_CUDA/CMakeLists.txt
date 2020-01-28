cmake_minimum_required(VERSION 3.8)
project(p33)

find_package(CUDA REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

include_directories(
        src
)

set(SOURCE_FILES
        src/complex.h
        src/complex.cc
        src/input_image.h
        src/input_image.cc
        )

cuda_add_executable(${PROJECT_NAME} ${SOURCE_FILES} main.cu)
