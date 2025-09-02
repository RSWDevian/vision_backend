# minimum version of CMake required Vision
cmake_minimum_required(VERSION 3.10)
#set the name of the project
project(Vision)

set(CMAKE_CXX_STANDARD 17)

# Find packages
find_package(OpenCV REQUIRED) 

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include/)

# Create detector library (only if sources exist)
file(GLOB DETECTOR_SOURCES "src/detector/*.cpp")
if(DETECTOR_SOURCES)
    # Remove detector_main.cpp from library sources if it exists
    list(FILTER DETECTOR_SOURCES EXCLUDE REGEX ".*detector_main\\.cpp$")
    if(DETECTOR_SOURCES)
        add_library(detector ${DETECTOR_SOURCES})
        target_include_directories(detector PUBLIC include/detector ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(detector ${OpenCV_LIBS})
    endif()
endif()

# Main Vision executable
add_executable(Vision
    src/main.cpp
    src/stereo.cpp
    src/pose.cpp
    src/calibration.cpp
    src/results.cpp
)
target_include_directories(Vision PRIVATE include/)
target_link_libraries(Vision ${OpenCV_LIBS})

# Test executables
add_executable(test_findChessboardCorners
    src/testers/calibration_testers/test_findChessboardCorners.cpp
    src/calibration.cpp
    src/stereo.cpp
)
target_include_directories(test_findChessboardCorners PRIVATE include/)
target_link_libraries(test_findChessboardCorners ${OpenCV_LIBS})

add_executable(test_depthMapping
    src/testers/stereo_testers/test_depthMapping.cpp
    src/stereo.cpp
)
target_include_directories(test_depthMapping PRIVATE include/)
target_link_libraries(test_depthMapping ${OpenCV_LIBS})

# Detector executable (only if detector_main.cpp exists)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/detector/detector_main.cpp" AND DETECTOR_SOURCES)
    add_executable(detector_main src/detector/detector_main.cpp)
    target_include_directories(detector_main PRIVATE include/detector)
    target_link_libraries(detector_main detector ${OpenCV_LIBS})
endif()
