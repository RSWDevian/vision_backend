#ifndef STEREO_HPP
#define STEREO_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>

class StereoVision{
private:
    //camera calibration data
    cv::Mat cameraMatrix1, cameraMatrix2;
    cv::Mat distCoeffs1, distCoeffs2;
    cv::Mat R, T, E, F;

    // Rectifications maps
    cv::Mat R1, R2, P1, P2, Q;
    cv::Mat map1x, map1y, map2x, map2y;

    //System state
    bool isCalibrated;
    cv::Size imageSize;

public:
    //Constructor & Destructor
    StereoVision();
    ~StereoVision();

    // Calibration methods
    bool calibrateStereo(
        const std::vector<std::vector<cv::Point3f>>& objectPoints,
        const std::vector<std::vector<cv::Point2f>>& imagePoints1,
        const std::vector<std::vector<cv::Point2f>>& imagePoints2,
        const cv::Size& imgSize 
    );
    bool loadCalibration(const std::string& filename);
    bool saveCalibration(const std::string& filename);

    // Rectification
    bool rectifyImages(const cv::Mat& left, const cv::Mat& right, cv::Mat& leftRect, cv::Mat& rightRect);
    
    // Disparity & Depth
    cv::Mat computeDisparityBM(const cv::Mat& leftRect, const cv::Mat& rightRect);
    cv::Mat computeDisparitySBGM(const cv::Mat& leftRect, const cv::Mat& rightRect);
    cv::Mat computeDepthMap(const cv::Mat& disparity);

    // Occlusion handing
    cv::Mat filterOcculsions(const cv::Mat& disparity);

    // Utility methods
    cv::Point3f reprojectTo3D(const cv::Point2f& point, float disparity);
    
    // Getter for calibration status
    bool isSystemCalibrated() const { return isCalibrated; }

};

#endif
