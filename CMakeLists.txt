cmake_minimum_required(VERSION 3.8.1)
project(robonet)


set(CMAKE_CXX_STANDARD 14)

add_subdirectory(clBP)
add_subdirectory(iir1)

find_package(OpenCV REQUIRED)

find_package(Boost REQUIRED)

add_executable(robonet main.cpp serialib.cpp neural.cpp neural.h)
target_link_libraries(robonet clBP ${OpenCV_LIBS} iir)
target_include_directories(robonet PRIVATE iir1)
