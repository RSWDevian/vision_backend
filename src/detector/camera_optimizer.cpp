#include "camera_optimizer.hpp"
#include <cmath>
#include <algorithm>
#include <iostream> 

//? Constructor
CameraOptimizer::CameraOptimizer(double baseLine_m)
    :baseline_m_(baseLine_m), baseline_mm_(baseLine_m * 1000.0){}

CameraProperties CameraOptimizer::optimizeForDistance(
    double target_distance_m,
    double min_player_height_px,
    double max_fov_deg
){
    CameraProperties optimal;

    // Standard Resolutions to test
    std::vector<cv::Size> resolutions = {
        cv::Size(1920, 1080),
        cv::Size(2560, 1440),
        cv::Size(3840, 2160),
        cv::Size(7680, 4320)
    };

    
}