cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

list(APPEND CMAKE_PREFIX_PATH "/home/luber90/libtorch")
project(cmag CXX CUDA)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${OPENCV_CXX_FLAGS}")

add_executable(networkc main.cpp unet.cpp)
target_link_libraries(networkc ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_property(TARGET networkc PROPERTY CXX_STANDARD 20)