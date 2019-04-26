cmake_minimum_required(VERSION 3.8.1)
project(robonet)


include_directories("/home/bruno/attempt2/pytorch/torch/include/torch/csrc/api/include/")

include_directories("/home/bruno/attempt2/pytorch/torch/include/")

set(CMAKE_CXX_STANDARD 14)

set(Caffe2_DIR "/home/bruno/attempt2/pytorch/torch/share/cmake/Caffe2/")

add_subdirectory(clBP)
add_subdirectory(iir1)
find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)

find_package(Boost REQUIRED)

set(CMAKE_CXX_STANDARD, 17)
add_executable(robonet main.cpp serialib.cpp neural.cpp neural.h)
#target_include_directories(robonet PRIVATE "C:\\Users\\Callum\\libtorch\\include\\torch\\csrc\\api\\include")
#target_include_directories(robonet PRIVATE "C:\\Users\\Callum\\libtorch\\include")
target_link_libraries(robonet clBP ${OpenCV_LIBS} ${TORCH_LIBRARIES} iir)
