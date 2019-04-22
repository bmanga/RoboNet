cmake_minimum_required(VERSION 3.8.1)
project(robonet)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(clBP)

find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD, 17)
add_executable(robonet main.cpp serialib.cpp neural.cpp neural.h)
#target_include_directories(robonet PRIVATE "C:\\Users\\Callum\\libtorch\\include\\torch\\csrc\\api\\include")
target_include_directories(robonet PRIVATE "C:\\Users\\Callum\\libtorch\\include")
target_link_libraries(robonet clBP ${OpenCV_LIBS} ${TORCH_LIBRARIES})
