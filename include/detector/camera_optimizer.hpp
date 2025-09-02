#ifndef CAMERA_OPTIMIZER_HPP
#define CAMERA_OPTIMIZER_HPP

#include <opencv2/core.hpp>
#include <vector>
#include <tuple>

//& Struct to store the camera properties
struct CameraProperties{
    double focal_length_px;
    cv::Size resolution;
    double fov_horizontal_deg;
    double fov_vertical_deg;
    double pixel_size_mm;
    double sensor_width_mm;
    double sensor_height_mm;
};

//& Struct to store the calculated depth accuracy
struct DepthAccuracy{
    double depth_resolution_cm_per_px;
    double min_measurable_depth_m;
    double max_measurable_depth_m;
    double disparity_range_px;
};

//& Required player visibility parameters
struct PlayerVisibility{
    double player_height_px;
    double player_width_px;
    bool is_processable;
    double detection_confidence;
};

//& Camera Optimzer class 
class CameraOptimizer{
private:
    double baseline_m_;
    double baseline_mm_;
    //? Helper functions
    double calculateDisparity(double focal_length_px, double distance_mm) const;
    double calculateDepthResolution(double focal_length_px, double distance_mm) const;
    double calculateMaxMeasurableDepth(double focal_length_px) const;

public:
    CameraOptimizer(double baseline_m);
    //? Function to caluclate optimal camera properties for given constraints
    CameraProperties optimizeForDistance(
        double target_disttance_m,
        double min_player_height_px = 100,
        double max_fov_deg = 120
    );

    //? Function to analyze depth accuracy for a given camera setup
    DepthAccuracy analyzeDepthAccuracy(
        const CameraProperties& camera,
        double distance_m
    ) const;

    //? Function to check player visibility at different distances
    PlayerVisibility analyzePlayerVisibility(
        const CameraProperties& camera,
        double distance_m,
        double player_height_m = 1.8
    ) const;

    //? Function to find optimal focal length range
    std::pair<double,double> findOptimalFocalLengthRange(
        double min_distance_m,
        double max_distance_m,
        const cv::Size& resolution
    );

    //? Calculate field of view from focal length
    double calculateHorizontalFOV(double focal_length_px, const cv::Size& resolution) const;
    double calculateVerticalFOV(double focal_length_px, const cv::Size& resolution) const;
    
    //? Generate camera recommendations
    std::vector<CameraProperties> generateRecommendations(double target_distance_m,
                                                         const std::vector<cv::Size>& resolutions); 
};


#endif // CAMERA_OPTIMIZER_HPP