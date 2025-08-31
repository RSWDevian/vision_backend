# minimum version of CMake required Vision
cmake_minimum_required(VERSION 3.10)
#set the name of the project
project(Vision)

set(CMAKE_CXX_STANDARD 14)

# Find packages
find_package(OpenCV REQUIRED) 

# Executables
add_executable(Vision
    src/main.cpp
    src/stereo.cpp
    src/pose.cpp
    src/calibration.cpp
    src/results.cpp
)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
target_include_directories(Vision PRIVATE include/)

# link libraries
target_link_libraries(Vision ${OpenCV_LIBS}) 
