#ifndef FIELD_ANALYZER_HPP
#define FIELD_ANALYZER_HPP

#include <opencv2/opencv.hpp>
#include <vector>

//& Struct to store the field dimensions and parameters
struct FieldDimensions
{
    double length_m;           // Total field length
    double width_m;            // Total field width
    double penalty_length_m;   // Penalty box length
    double penalty_width_m;    // Penalty box width
    double goal_width_m;       // Goal width
    double goal_area_length_m; // Goal area length
    double goal_area_width_m;  // Goal area width
};

//& Struct to store player zone information
struct PlayerZone
{
    cv::Point2d center; // Center of zone in field coordinates
    double radius_m;    // Radius of zone
    std::string name;   // Zone name (e.g., "penalty_box", "center_circle")
    int priority;       // Priority for camera optimization (1-10)
};

//& Struct to store coverage area information
struct CoverageArea
{
    std::vector<cv::Point2d> polygon; // Area boundary in field coordinates
    double min_distance_m;            // Minimum distance to any point
    double max_distance_m;            // Maximum distance to any point
    double avg_distance_m;            // Average distance to area
    std::string description;          // Area description
};

//& Class for analyzing field dimensions and player zones
class FieldAnalyzer
{
private:
    FieldDimensions field_dims_;

    // Helper functions
    double calculateDistance(const cv::Point2d &p1, const cv::Point2d &p2) const;
    bool pointInPolygon(const cv::Point2d &point, const std::vector<cv::Point2d> &polygon) const;
    std::vector<cv::Point2d> generateFieldBoundary() const;

public:
    FieldAnalyzer();

    //? Set field dimensions (default FIFA standard)
    void setFieldDimensions(const FieldDimensions &dimensions);
    FieldDimensions getStandardFieldDimensions() const;

    //? Define important zones for camera optimization
    std::vector<PlayerZone> definePlayerZones() const;

    //? Calculate coverage areas from camera positions
    CoverageArea calculateCoverageArea(const cv::Point2d &camera_pos, double fov_deg, double max_range_m) const;

    //? Find optimal viewing positions for penalty box coverage
    std::vector<cv::Point2d> findOptimalViewingPositions(double baseline_m, double min_distance_m = 20.0) const;

    //? Calculate distances from position to key field areas
    std::vector<double> calculateDistancesToKeyAreas(const cv::Point2d &position) const;

    //? Analyze field coverage for stereo pair
    double analyzeStereoCoverage(const cv::Point2d &left_camera, const cv::Point2d &right_camera,
                                 double fov_deg, double max_range_m) const;

    //? Get penalty box corners in field coordinates
    std::vector<cv::Point2d> getPenaltyBoxCorners() const;

    //? Convert between field coordinates and camera coordinates
    cv::Point2d fieldToCamera(const cv::Point2d &field_point, const cv::Point2d &camera_pos, double camera_angle_deg = 0) const;

    //? Validate camera positions for practical constraints
    bool validateCameraPosition(const cv::Point2d &position, double safety_margin_m = 5.0) const;
};

#endif // FIELD_ANALYZER_HPP