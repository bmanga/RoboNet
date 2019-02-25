cmake_minimum_required(VERSION 3.8.1)
project(robonet)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(/home/sama/Documents/clBP /home/sama/Documents/clBP/cmake-build-debug)

find_package(OpenCV REQUIRED)

add_executable(robonet main.cpp)
target_link_libraries(robonet clBP ${OpenCV_LIBS})