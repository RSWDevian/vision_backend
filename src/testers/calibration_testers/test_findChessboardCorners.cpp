#include "calibration.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>

class ChessboardCornersTester{
private:
    cv::Size boardSize;
    float squareSize;
public:
    //? Constructor
    ChessboardCornersTester(
        cv::Size board_Size,
        float square_Size
    ):boardSize(board_Size), squareSize(square_Size) {}

    //? Test the findChessboardCorners function timeout
    bool testFindChessboardCornersWithTimeout(
        const cv::Mat& image,
        std::vector<cv::Point2f>& corners,
        int timeoutSeconds = 10
    ){
        std::cout<<"Testing findChessboardCorners function..."<<std::endl;
        std::cout<<"Image Size:"<<image.size()<<std::endl;
        std::cout<<"Image channels:"<<image.channels()<<std::endl;
        std::cout<<"Board Size"<<boardSize<<std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        // convert to grayscale if needed
        cv::Mat gray;
        if(image.channels() == 3){
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            std::cout << "Converted to grayscale" << std::endl;
        }
        else {
            gray = image.clone();
            std::cout << "Image already grayscale" << std::endl;
        }

        // Test different flag combinations
        std::vector<std::pair<std::string, int>> flagCombinations = {
            {"ADAPTIVE_THRESH | NORMALIZE_IMAGE | FAST_CHECK", 
             cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK},
            {"ADAPTIVE_THRESH | NORMALIZE_IMAGE", 
             cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE},
            {"ADAPTIVE_THRESH | FAST_CHECK", 
             cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK},
            {"ADAPTIVE_THRESH only", 
             cv::CALIB_CB_ADAPTIVE_THRESH},
            {"NORMALIZE_IMAGE only", 
             cv::CALIB_CB_NORMALIZE_IMAGE},
            {"FAST_CHECK only", 
             cv::CALIB_CB_FAST_CHECK},
            {"No flags", 0}
        };

        bool foundAny = false;

        for (const auto& [flagName, flags] : flagCombinations) {
            std::cout << "\n--- Testing with flags: " << flagName << " ---" << std::endl;
            
            auto testStart = std::chrono::high_resolution_clock::now();
            
            corners.clear();
            bool found = cv::findChessboardCorners(gray, boardSize, corners, flags);
            
            auto testEnd = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(testEnd - testStart);
            
            std::cout << "Result: " << (found ? "FOUND" : "NOT FOUND") << std::endl;
            std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
            std::cout << "Corners detected: " << corners.size() << std::endl;
            
            if (found) {
                foundAny = true;
                std::cout << "✓ Success with flags: " << flagName << std::endl;
                
                // Test corner refinement
                auto refineStart = std::chrono::high_resolution_clock::now();
                cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
                auto refineEnd = std::chrono::high_resolution_clock::now();
                auto refineDuration = std::chrono::duration_cast<std::chrono::milliseconds>(refineEnd - refineStart);
                
                std::cout << "Corner refinement took: " << refineDuration.count() << " ms" << std::endl;
                break; // Stop at first successful detection
            }
            
            // Check if we're taking too long
            auto current = std::chrono::high_resolution_clock::now();
            auto totalDuration = std::chrono::duration_cast<std::chrono::seconds>(current - start);
            if (totalDuration.count() > timeoutSeconds) {
                std::cout << "\n⚠️ TIMEOUT: Test exceeded " << timeoutSeconds << " seconds" << std::endl;
                return false;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "\nTotal test time: " << totalDuration.count() << " ms" << std::endl;
        
        return foundAny;
    }

    // Test with image preprocessing
    bool testWithPreprocessing(const cv::Mat& image, std::vector<cv::Point2f>& corners) {
        std::cout << "\n=== Testing with preprocessing ===" << std::endl;
        
        cv::Mat processed;
        if (image.channels() == 3) {
            cv::cvtColor(image, processed, cv::COLOR_BGR2GRAY);
        } else {
            processed = image.clone();
        }
        
        // Test different preprocessing techniques
        std::vector<std::pair<std::string, cv::Mat>> preprocessedImages;
        
        // Original
        preprocessedImages.push_back({"Original", processed.clone()});
        
        // Gaussian blur
        cv::Mat blurred;
        cv::GaussianBlur(processed, blurred, cv::Size(5, 5), 0);
        preprocessedImages.push_back({"Gaussian Blur", blurred});
        
        // Histogram equalization
        cv::Mat equalized;
        cv::equalizeHist(processed, equalized);
        preprocessedImages.push_back({"Histogram Equalized", equalized});
        
        // Bilateral filter
        cv::Mat bilateral;
        cv::bilateralFilter(processed, bilateral, 9, 75, 75);
        preprocessedImages.push_back({"Bilateral Filter", bilateral});
        
        // Contrast adjustment
        cv::Mat contrast;
        processed.convertTo(contrast, -1, 1.5, 0); // alpha=1.5 (contrast), beta=0 (brightness)
        preprocessedImages.push_back({"Enhanced Contrast", contrast});
        
        bool foundAny = false;
        for (const auto& [name, img] : preprocessedImages) {
            std::cout << "\n--- Testing with: " << name << " ---" << std::endl;
            
            corners.clear();
            auto start = std::chrono::high_resolution_clock::now();
            
            bool found = cv::findChessboardCorners(img, boardSize, corners,
                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            std::cout << "Result: " << (found ? "FOUND" : "NOT FOUND") << std::endl;
            std::cout << "Time: " << duration.count() << " ms" << std::endl;
            std::cout << "Corners: " << corners.size() << std::endl;
            
            if (found) {
                foundAny = true;
                std::cout << "✓ Success with preprocessing: " << name << std::endl;
            }
        }
        
        return foundAny;
    }

    // Visualize detection results
    void visualizeResults(const cv::Mat& image, const std::vector<cv::Point2f>& corners, 
                         bool found, const std::string& windowName = "Chessboard Detection") {
        cv::Mat visualization = image.clone();
        
        if (found && !corners.empty()) {
            cv::drawChessboardCorners(visualization, boardSize, corners, found);
            std::cout << "✓ Visualization created with detected corners" << std::endl;
        } else {
            cv::putText(visualization, "No chessboard detected", cv::Point(50, 50),
                       cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            std::cout << "✗ No corners to visualize" << std::endl;
        }
        
        // Save the visualization
        std::string filename = windowName + "_result.png";
        cv::imwrite(filename, visualization);
        std::cout << "Visualization saved as: " << filename << std::endl;
        
        cv::imshow(windowName, visualization);
        cv::waitKey(0);
        cv::destroyWindow(windowName);
    }
};

//? Test function that uses the Calibrator class
bool testCalibratorFindChessboardCorners(const std::string& imagePath, 
                                        cv::Size boardSize, 
                                        float squareSize) {
    std::cout << "\n=== Testing Calibrator::findChessboardCorners ===" << std::endl;
    std::cout << "Image: " << imagePath << std::endl;
    
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "✗ Could not load image: " << imagePath << std::endl;
        return false;
    }
    
    Calibrator calibrator(boardSize, squareSize);
    std::vector<cv::Point2f> corners;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::cout << "Calling Calibrator::findChessboardCorners..." << std::endl;
    bool found = calibrator.findChessboardCorners(image, corners);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Result: " << (found ? "FOUND" : "NOT FOUND") << std::endl;
    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
    std::cout << "Corners detected: " << corners.size() << std::endl;
    
    // Create visualization
    cv::Mat result = calibrator.drawChessBoardCorners(image, corners);
    std::string outputFile = "calibrator_test_result.png";
    cv::imwrite(outputFile, result);
    std::cout << "Result saved as: " << outputFile << std::endl;
    
    return found;
}

//& main function for the Calibration tester
int main() {
    std::cout << "=== Chessboard Corner Detection Tester ===" << std::endl;
    
    // Configuration
    cv::Size boardSize(7, 7);  // Adjust to match your chessboard
    float squareSize = 25.0f;  // Adjust to match your square size
    
    // Test images paths - adjust to your actual test images
    std::vector<std::string> testImages = {
        "../testingImages/Calibration_Samples/calibrate1/LeftCamera.png",
        "../testingImages/Calibration_Samples/calibrate1/RightCamera.png",
        "../testingImages/Calibration_Samples/calibrate2/LeftCamera.png",
        "../testingImages/Calibration_Samples/calibrate2/RightCamera.png"
    };
    
    ChessboardCornersTester tester(boardSize, squareSize);
    
    for (const std::string& imagePath : testImages) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Testing image: " << imagePath << std::endl;
        
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cout << "✗ Could not load image: " << imagePath << std::endl;
            continue;
        }
        
        std::vector<cv::Point2f> corners;
        
        // Test 1: Basic detection with timeout
        bool found1 = tester.testFindChessboardCornersWithTimeout(image, corners, 10);
        
        // Test 2: With preprocessing
        bool found2 = tester.testWithPreprocessing(image, corners);
        
        // Test 3: Using Calibrator class
        bool found3 = testCalibratorFindChessboardCorners(imagePath, boardSize, squareSize);
        
        // Visualize the best result
        if (found1 || found2 || found3) {
            tester.visualizeResults(image, corners, found1 || found2 || found3, 
                                  "Test_" + std::filesystem::path(imagePath).stem().string());
        }
        
        std::cout << "\nSummary for " << imagePath << ":" << std::endl;
        std::cout << "  Direct OpenCV test: " << (found1 ? "✓" : "✗") << std::endl;
        std::cout << "  Preprocessing test: " << (found2 ? "✓" : "✗") << std::endl;
        std::cout << "  Calibrator test: " << (found3 ? "✓" : "✗") << std::endl;
    }
    
    std::cout << "\n=== Testing Complete ===" << std::endl;
    std::cout << "Check the generated result images for visual verification." << std::endl;
    
    return 0;
}

