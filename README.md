#include "../include/calibration.hpp"
#include "../include/stereo.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

// Constructor
Calibrator::Calibrator(cv::Size board_Size, float square_Size) 
    : boardSize(board_Size), squareSize(square_Size) {
    stereo = new StereoVision();
}

// Destructor
Calibrator::~Calibrator() {
    delete stereo;
}

// Generate 3D chessboard corner points
std::vector<cv::Point3f> Calibrator::generateChessboardPoints() {
    std::vector<cv::Point3f> corners;
    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            corners.push_back(cv::Point3f(j * squareSize, i * squareSize, 0));
        }
    }
    return corners;
}

// Find chessboard corners in image
bool Calibrator::findChessboardCorners(const cv::Mat& image, std::vector<cv::Point2f>& corners) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    bool found = cv::findChessboardCorners(gray, boardSize, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);

    if (found) {
        // Refine corner positions for better accuracy
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
    }

    return found;
}

// Add calibration image pair
bool Calibrator::addCalibrationImages(const cv::Mat& leftImage, const cv::Mat& rightImage) {
    std::vector<cv::Point2f> cornersLeft, cornersRight;

    std::cout << "Processing calibration image pair..." << std::endl;
    
    bool foundLeft = findChessboardCorners(leftImage, cornersLeft);
    bool foundRight = findChessboardCorners(rightImage, cornersRight);

    if (foundLeft && foundRight) {
        imagePointsLeft.push_back(cornersLeft);
        imagePointsRight.push_back(cornersRight);
        objectPoints.push_back(generateChessboardPoints());
        
        std::cout << "✓ Chessboard detected in both images. Total pairs: " 
                  << objectPoints.size() << std::endl;
        return true;
    } else {
        std::cout << "✗ Chessboard detection failed: ";
        if (!foundLeft) std::cout << "Left image ";
        if (!foundRight) std::cout << "Right image ";
        std::cout << std::endl;
        return false;
    }
}

// Perform calibration using your StereoVision class
bool Calibrator::performCalibration(const cv::Size& imageSize) {
    if (objectPoints.size() < 3) {
        std::cout << "Need at least 3 calibration image pairs. Current: " 
                  << objectPoints.size() << std::endl;
        return false;
    }

    std::cout << "\n=== Starting Stereo Calibration ===" << std::endl;
    std::cout << "Using " << objectPoints.size() << " image pairs" << std::endl;
    std::cout << "Board size: " << boardSize.width << "x" << boardSize.height << std::endl;
    std::cout << "Square size: " << squareSize << "mm" << std::endl;
    std::cout << "Image size: " << imageSize.width << "x" << imageSize.height << std::endl;

    // Use your StereoVision calibrateStereo function
    bool success = stereo->calibrateStereo(objectPoints, imagePointsLeft, 
                                         imagePointsRight, imageSize);
    
    if (success) {
        std::cout << "✓ Calibration completed successfully!" << std::endl;
    } else {
        std::cout << "✗ Calibration failed!" << std::endl;
    }

    return success;
}

// Save calibration using your StereoVision class
bool Calibrator::saveCalibrationData(const std::string& filename) {
    bool success = stereo->saveCalibration(filename);
    if (success) {
        std::cout << "✓ Calibration data saved to: " << filename << std::endl;
    } else {
        std::cout << "✗ Failed to save calibration data" << std::endl;
    }
    return success;
}

// Load calibration using your StereoVision class
bool Calibrator::loadCalibrationData(const std::string& filename) {
    bool success = stereo->loadCalibration(filename);
    if (success) {
        std::cout << "✓ Calibration data loaded from: " << filename << std::endl;
    } else {
        std::cout << "✗ Failed to load calibration data" << std::endl;
    }
    return success;
}

// Get the StereoVision object
StereoVision* Calibrator::getStereoSystem() {
    return stereo;
}

// Draw chessboard corners for visualization
cv::Mat Calibrator::drawChessBoardCorners(const cv::Mat& image, std::vector<cv::Point2f>& corners) {
    cv::Mat result = image.clone();
    cv::drawChessboardCorners(result, boardSize, corners, true);
    return result;
}

// Print calibration statistics
void Calibrator::printCalibrationStats() {
    std::cout << "\n=== Calibration Statistics ===" << std::endl;
    std::cout << "Board size: " << boardSize.width << "x" << boardSize.height << std::endl;
    std::cout << "Square size: " << squareSize << " mm" << std::endl;
    std::cout << "Number of calibration image pairs: " << objectPoints.size() << std::endl;
    std::cout << "System calibrated: " << (stereo->isSystemCalibrated() ? "Yes" : "No") << std::endl;
}

// Calibration Workflow Implementation
namespace CalibrationWorkflow {
    
    // Main function to calibrate from your Calibration_Samples folder
    bool calibrateFromTestImages() {
        std::cout << "\n=== Stereo Camera Calibration from Calibration Samples ===" << std::endl;
        
        // Chessboard parameters - adjust based on your actual chessboard
        cv::Size boardSize(9, 6);  // 9x6 internal corners - measure your chessboard
        float squareSize = 25.0f;  // 25mm squares - measure your actual chessboard squares
        
        Calibrator calibrator(boardSize, squareSize);
        
        // Process all 5 calibration sample folders
        std::vector<std::string> sampleFolders = {
            "testingImages/Calibration_Samples/calibrate1/",
            "testingImages/Calibration_Samples/calibrate2/",
            "testingImages/Calibration_Samples/calibrate3/",
            "testingImages/Calibration_Samples/calibrate4/",
            "testingImages/Calibration_Samples/calibrate5/"
        };
        
        int successfulPairs = 0;
        cv::Size imageSize;
        
        for (int i = 0; i < sampleFolders.size(); i++) {
            std::cout << "\n--- Processing " << sampleFolders[i] << " ---" << std::endl;
            
            // Handle different naming patterns in your folders
            std::vector<std::pair<std::string, std::string>> possibleNames = {
                {"LeftCamera.png", "RightCamera.png"},
                {"LaftCamera.png", "RightCamera.png"},  // Handle typo in calibrate1
                {"left.png", "right.png"},
                {"Left.png", "Right.png"}
            };
            
            bool pairFound = false;
            for (auto& names : possibleNames) {
                std::string leftPath = sampleFolders[i] + names.first;
                std::string rightPath = sampleFolders[i] + names.second;
                
                cv::Mat leftImage = cv::imread(leftPath);
                cv::Mat rightImage = cv::imread(rightPath);
                
                if (!leftImage.empty() && !rightImage.empty()) {
                    std::cout << "✓ Loaded: " << names.first << " & " << names.second << std::endl;
                    std::cout << "Image size: " << leftImage.size() << std::endl;
                    
                    // Store image size from first successful load
                    if (imageSize.width == 0) {
                        imageSize = leftImage.size();
                    }
                    
                    // Verify both images have same size
                    if (leftImage.size() != rightImage.size()) {
                        std::cout << "✗ Warning: Left and right images have different sizes!" << std::endl;
                        continue;
                    }
                    
                    // Add to calibration
                    if (calibrator.addCalibrationImages(leftImage, rightImage)) {
                        successfulPairs++;
                    }
                    pairFound = true;
                    break;
                }
            }
            
            if (!pairFound) {
                std::cout << "✗ Could not load images from " << sampleFolders[i] << std::endl;
                
                // Debug: List actual files in the folder
                std::cout << "Debug: Looking for files in folder..." << std::endl;
                // You could add directory listing here if needed
            }
        }
        
        std::cout << "\n=== Calibration Summary ===" << std::endl;
        std::cout << "Processed folders: " << sampleFolders.size() << std::endl;
        std::cout << "Successful calibration pairs: " << successfulPairs << std::endl;
        
        if (successfulPairs < 3) {
            std::cerr << "✗ Need at least 3 successful calibration pairs!" << std::endl;
            std::cerr << "Check that your chessboard images contain detectable patterns." << std::endl;
            return false;
        }
        
        // Perform calibration using your StereoVision class
        if (!calibrator.performCalibration(imageSize)) {
            return false;
        }
        
        // Save calibration data
        std::string outputFile = "stereo_calibration.xml";
        if (!calibrator.saveCalibrationData(outputFile)) {
            return false;
        }
        
        // Print final statistics
        calibrator.printCalibrationStats();
        
        std::cout << "\n✓ Calibration workflow completed successfully!" << std::endl;
        std::cout << "Calibration data saved to: " << outputFile << std::endl;
        std::cout << "You can now use this calibration for stereo vision tasks." << std::endl;
        
        return true;
    }
    
    // Test loading existing calibration
    bool testLoadCalibration(const std::string& calibrationFile) {
        std::cout << "\n=== Testing Calibration Load ===" << std::endl;
        
        StereoVision stereo;
        bool success = stereo.loadCalibration(calibrationFile);
        
        if (success) {
            std::cout << "✓ Successfully loaded calibration from: " << calibrationFile << std::endl;
            std::cout << "System calibrated: " << (stereo.isSystemCalibrated() ? "Yes" : "No") << std::endl;
        } else {
            std::cout << "✗ Failed to load calibration from: " << calibrationFile << std::endl;
        }
        
        return success;
    }
    
    // Validate calibration images before processing
    bool validateCalibrationImages(const std::string& leftPath, const std::string& rightPath) {
        cv::Mat left = cv::imread(leftPath);
        cv::Mat right = cv::imread(rightPath);
        
        if (left.empty() || right.empty()) {
            return false;
        }
        
        // Check if images have the same size
        if (left.size() != right.size()) {
            std::cout << "Warning: Left and right images have different sizes!" << std::endl;
            return false;
        }
        
        return true;
    }
    
    // Display calibration results
    bool displayCalibrationResults(const StereoVision& stereo) {
        std::cout << "\n=== Calibration Results ===" << std::endl;
        std::cout << "System Status: " << (stereo.isSystemCalibrated() ? "CALIBRATED" : "NOT CALIBRATED") << std::endl;
        return stereo.isSystemCalibrated();
    }
}

// Utility functions implementation
namespace CalibrationUtils {
    
    // Detect chessboard corners
    bool detectChessboardCorners(const cv::Mat& image, cv::Size boardSize, std::vector<cv::Point2f>& corners) {
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }

        return cv::findChessboardCorners(gray, boardSize, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
    }
    
    // Get default calibration parameters
    CalibrationParams getDefaultCalibrationParams() {
        CalibrationParams params;
        return params;
    }
    
    // Validate parameters
    bool validateParams(const CalibrationParams& params) {
        if (params.boardSize.width <= 0 || params.boardSize.height <= 0) {
            std::cerr << "Invalid board size!" << std::endl;
            return false;
        }
        
        if (params.squareSize <= 0) {
            std::cerr << "Invalid square size!" << std::endl;
            return false;
        }
        
        return true;
    }
    
    // Check if calibration file exists
    bool checkCalibrationFileExists(const std::string& filename) {
        std::ifstream file(filename);
        return file.good();
    }
    
    // Visualize chessboard detection
    cv::Mat visualizationChessboardDetection(const cv::Mat& image, const cv::Size& boardSize,
                                            const std::vector<cv::Point2f>& corners, bool found) {
        cv::Mat result = image.clone();
        cv::drawChessboardCorners(result, boardSize, corners, found);
        return result;
    }
}

// Global functions for easy use in main.cpp
bool runStereoCalibration() {
    return CalibrationWorkflow::calibrateFromTestImages();
}

bool testCalibrationLoad(const std::string& filename) {
    return CalibrationWorkflow::testLoadCalibration(filename);
}