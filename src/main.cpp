#include <opencv2/opencv.hpp>
#include <iostream>
#include "../include/calibration.hpp"
#include "../include/stereo.hpp"

void printBaselineInfo() {
    cv::FileStorage fs("stereo_calibration.xml", cv::FileStorage::READ);
    
    if (fs.isOpened()) {
        cv::Mat T, Q;
        fs["T"] >> T;
        fs["Q"] >> Q;
        
        std::cout << "\n=== BASELINE FROM CALIBRATION FILE ===" << std::endl;
        
        // From Translation vector
        double baseline_T = T.at<double>(0,0);
        std::cout << "Baseline from T vector: " << baseline_T << " mm (" 
                  << baseline_T/1000.0 << " meters)" << std::endl;
        
        // From Q matrix
        double baseline_Q = -1.0 / Q.at<double>(3, 2);
        std::cout << "Baseline from Q matrix: " << baseline_Q << " mm (" 
                  << baseline_Q/1000.0 << " meters)" << std::endl;
        
        // Calculate effective baseline (absolute value)
        double effective_baseline = std::abs(baseline_T) / 1000.0;
        std::cout << "Effective baseline: " << effective_baseline << " meters" << std::endl;
        
        // Display Q matrix for reference
        std::cout << "\nQ Matrix:" << std::endl;
        std::cout << Q << std::endl;
        
        fs.release();
    } else {
        std::cout << "Could not read calibration file!" << std::endl;
    }
}

int main(){
    std::cout << "=== Stereo Vision Calibration Test ===" << std::endl;
    
    // Use the calibrateFromTestImages function from CalibrationWorkflow namespace
    std::cout << "\nStarting calibration from Calibration_Samples folder..." << std::endl;
    
    bool calibrationSuccess = CalibrationWorkflow::calibrateFromTestImages();
    
    if (calibrationSuccess) {
        std::cout << "\n✓ Calibration completed successfully!" << std::endl;
        
        // Print baseline information from calibration
        printBaselineInfo();
        
        // Test loading the calibration to verify it was saved correctly
        std::cout << "\nTesting calibration load..." << std::endl;
        bool loadSuccess = CalibrationWorkflow::testLoadCalibration("stereo_calibration.xml");
        
        if (loadSuccess) {
            std::cout << "\n✓ Calibration file verified!" << std::endl;
            
            // Load and display some test images with the calibrated system
            std::cout << "\nTesting with sample images..." << std::endl;
            
            cv::Mat leftImg = cv::imread("../testingImages/testPair2/left_camera_view.png");
            cv::Mat rightImg = cv::imread("../testingImages/testPair2/right_camera_view.png");
            
            if (!leftImg.empty() && !rightImg.empty()) {
                std::cout << "✓ Test images loaded successfully" << std::endl;
                std::cout << "Image size: " << leftImg.size() << std::endl;
                
                // Create resizable windows
                cv::namedWindow("Left Image", cv::WINDOW_NORMAL);
                cv::namedWindow("Right Image", cv::WINDOW_NORMAL);
                
                // Resize windows to a smaller size (width, height)
                cv::resizeWindow("Left Image", 640, 480);
                cv::resizeWindow("Right Image", 640, 480);
                
                // Position windows side by side
                cv::moveWindow("Left Image", 100, 100);
                cv::moveWindow("Right Image", 750, 100);

                cv::imshow("Left Image", leftImg);
                cv::imshow("Right Image", rightImg);
                
                std::cout << "\nPress any key to close windows..." << std::endl;
                cv::waitKey(0);
                cv::destroyAllWindows();
            } else {
                std::cout << "✗ Could not load test images for verification" << std::endl;
            }
            
        } else {
            std::cout << "✗ Failed to load saved calibration" << std::endl;
        }
        
    } else {
        std::cout << "✗ Calibration failed!" << std::endl;
        std::cout << "Check that your chessboard images are valid and contain detectable patterns." << std::endl;
    }

    return 0;
}