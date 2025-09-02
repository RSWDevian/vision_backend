#ifndef CALIBRATION_HPP
#define CALIBRATION_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "stereo.hpp"

class StereoVision;

class Calibrator{
private:
    cv::Size boardSize;
    float squareSize;
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePointsLeft;
    std::vector<std::vector<cv::Point2f>> imagePointsRight;
    StereoVision* stereo;

public:
    //? constructor and destructor functions
    Calibrator(cv::Size board_Size, float square_Size);
    ~Calibrator();

    //? Core Calibration Methods
    std::vector<cv::Point3f> generateChessboardPoints();
    bool findChessboardCorners(const cv::Mat& image, std::vector<cv::Point2f>& corners);
    bool addCalibrationImages(const cv::Mat& leftImage, const cv::Mat& rightImage);
    bool performCalibration(const cv::Size& imageSize);

    // File I/O methods
    bool saveCalibrationData(const std::string& filename);
    bool loadCalibrationData(const std::string& filename);

    // Utility methods
    StereoVision* getStereoSystem();
    cv::Mat drawChessBoardCorners(const cv::Mat& image, std::vector<cv::Point2f>& corners);
    void printCalibrationStats();
};

namespace CalibrationWorkflow{
    // Main calibration functions
    bool calibrateFromTestImages();
    bool testLoadCalibration(const std::string& calibrationFile);

    // Helper funstions
    bool validateCalibrationImages(const std::string& leftPath, const std::string& rightPath);
    bool displayCalibrationResults(const StereoVision& stereo);
}

namespace CalibrationUtils{
    // chess board detection utilities
    bool detectChessboardCorners(
        const cv::Mat& image,
        cv::Size boardSize,
        std::vector<cv::Point2f>& corners
    );

    // Calibration Validation
    bool validateCalibrationQuality(
        double rms,
        const cv::Mat& cameraMatrixL,
        const cv::Mat& cameraMatrixR
    );

    // Visualization Helpers
    cv::Mat visualizationChessboardDetection(
        const cv::Mat& image,
        const cv::Size& boardSize,
        const std::vector<cv::Point2f>& corners,
        bool found
    );

    // File operations
    bool checkCalibrationFileExists(const std::string& filename);
    bool backupCalibrationFile(const std::string& filename);

    // Calibration parameters
    struct CalibrationParams{
        cv::Size boardSize;
        float squareSize;
        int minCalibrationImages;
        double maxAcceptableRMS;
        CalibrationParams():
            boardSize(cv::Size(9, 6)),
            squareSize(25.0f),
            minCalibrationImages(10),
            maxAcceptableRMS(1.0)
        {}
    };

    CalibrationParams getDefaultCalibrationParams();
    bool validateParams(const CalibrationParams& params);
}

//? Global calibration functions that can be called from main.cpp
bool runStereoCalibration();
bool testCalibrationLoad(const std::string& filename = "stereo_calibration.xml");

//? Advanced calibration features
namespace AdvancedCalibration {
    // Multi-image calibration
    bool calibrateFromImageDirectory(const std::string& leftDir, const std::string& rightDir,
                                   const std::string& outputFile);
    
    // Real-time calibration
    bool liveCalibrationCapture(int leftCameraId = 0, int rightCameraId = 1);
    
    // Calibration quality assessment
    struct CalibrationQuality {
        double stereoRMS;
        double leftCameraRMS;
        double rightCameraRMS;
        double epipolarError;
        bool isGoodCalibration;
    };
    
    CalibrationQuality assessCalibrationQuality(const StereoVision& stereo,
                                              const std::vector<std::vector<cv::Point3f>>& objectPoints,
                                              const std::vector<std::vector<cv::Point2f>>& leftPoints,
                                              const std::vector<std::vector<cv::Point2f>>& rightPoints);
}

#endif // CALIBRATION_HPP