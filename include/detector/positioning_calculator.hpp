#ifndef POSITIONING_CALCULATOR_HPP
#define POSITIONING_CALCULATOR_HPP

#include "camera_optimizer.hpp"
#include "field_analyzer.hpp"
#include <vector>

//& Struct to store camera position information
struct CameraPosition
{
    cv::Point2d location; // X,Y coordinates in field system
    double height_m;      // Camera height above ground
    double tilt_deg;      // Tilt angle (0 = horizontal)
    double pan_deg;       // Pan angle (0 = facing +X direction)
    std::string label;    // Position identifier
};

//& Struct to store stereo camera setup
struct StereoSetup
{
    CameraPosition left_camera;
    CameraPosition right_camera;
    double baseline_m;
    double convergence_distance_m; // Distance where cameras converge
    cv::Point2d midpoint;          // Midpoint between cameras
};

//& Struct to store optimization results
struct OptimizationResult
{
    StereoSetup stereo_setup;
    CameraProperties camera_props;
    DepthAccuracy depth_accuracy;
    PlayerVisibility player_visibility;
    double coverage_score; // 0-1 quality score
    std::string analysis;  // Text analysis
};

//& Class for calculating optimal camera positions
class PositioningCalculator
{
private:
    const FieldAnalyzer &field_analyzer_;
    const CameraOptimizer &camera_optimizer_;

    // Helper functions
    double calculateCoverageScore(const StereoSetup &setup, const CameraProperties &camera_props);
    double calculateConvergenceAngle(const StereoSetup &setup, double target_distance_m);
    std::string generateAnalysisText(const OptimizationResult &result);
    bool isPositionPractical(const CameraPosition &position);

public:
    PositioningCalculator(const FieldAnalyzer &field_analyzer, const CameraOptimizer &camera_optimizer);

    // Find optimal stereo camera positions
    std::vector<OptimizationResult> findOptimalPositions(
        double baseline_m,
        const std::vector<double> &distance_options_m,
        const std::vector<cv::Size> &resolution_options);

    // Calculate specific positioning for given constraints
    StereoSetup calculateStereoSetup(
        double baseline_m,
        double distance_from_penalty_m,
        double camera_height_m = 3.0);

    // Analyze existing setup
    OptimizationResult analyzeSetup(const StereoSetup &setup, const CameraProperties &camera_props);

    // Generate positioning recommendations
    std::vector<std::string> generatePositioningRecommendations(const OptimizationResult &result);

    // Calculate camera angles for optimal penalty box coverage
    std::pair<double, double> calculateOptimalAngles(const CameraPosition &camera_pos);

    // Validate stereo geometry
    bool validateStereoGeometry(const StereoSetup &setup, double min_convergence_angle_deg = 5.0);

    // Calculate effective baseline at target distance
    double calculateEffectiveBaseline(const StereoSetup &setup, double target_distance_m);

    // Find compromise between depth accuracy and player visibility
    OptimizationResult findOptimalCompromise(
        double baseline_m,
        double min_player_height_px = 100,
        double max_depth_error_cm = 50);
};

#endif // POSITIONING_CALCULATOR_HPP