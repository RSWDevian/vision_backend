#include "stereo.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>

class DepthMappingTester
{
private:
    StereoVision *stereoSystem;
    std::string calibrationFile;

public:
    //? Depth mapping tester
    DepthMappingTester(const std::string &caliFile) : calibrationFile(caliFile)
    {
        stereoSystem = new StereoVision();
    }

    ~DepthMappingTester()
    {
        delete stereoSystem;
    }

    //? Loading calibration
    bool loadCalibration()
    {
        std::cout << "Loading calibration from: " << calibrationFile << std::endl;
        bool success = stereoSystem->loadCalibration(calibrationFile);
        if (success)
        {
            std::cout << "✓ Calibration loaded successfully" << std::endl;
            std::cout << "System calibrated: " << (stereoSystem->isSystemCalibrated() ? "Yes" : "No") << std::endl;
        }
        else
        {
            std::cout << "✗ Failed to load calibration" << std::endl;
        }

        return success;
    }

    //? Function to load test images
    bool loadTestImages(
        const std::string &leftPath,
        const std::string &rightPath,
        cv::Mat &leftImage,
        cv::Mat &rightImage)
    {
        std::cout << "\nLoading test images..." << std::endl;
        std::cout << "Left image path: " << leftImage << std::endl;
        std::cout << "Right image path: " << rightImage << std::endl;

        leftImage = cv::imread(leftPath);
        rightImage = cv::imread(rightPath);

        if (leftImage.empty())
        {
            std::cout << "✗ Could not load left image: " << leftPath << std::endl;
            return false;
        }
        if (rightImage.empty())
        {
            std::cout << "✗ Could not load right image: " << rightPath << std::endl;
            return false;
        }

        std::cout << "✓ Images loaded successfully" << std::endl;
        std::cout << "Left image size: " << leftImage.size() << std::endl;
        std::cout << "Right image size: " << rightImage.size() << std::endl;

        // Check if images have the same size
        if (leftImage.size() != rightImage.size())
        {
            std::cout << "⚠️ Warning: Left and right images have different sizes!" << std::endl;
            return false;
        }

        return true;
    }

    //? Function to test rectification
    bool testRectification(
        const cv::Mat &leftImage,
        const cv::Mat &rightImage,
        cv::Mat &leftRect,
        cv::Mat &rightRect)
    {
        std::cout << "\n=== Testing Image Rectification ===" << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        bool success = stereoSystem->rectifyImages(leftImage, rightImage, leftRect, rightRect);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        if (success)
        {
            std::cout << "✓ Rectification successful" << std::endl;
            std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
            std::cout << "Rectified image size: " << leftRect.size() << std::endl;

            // Save rectified images
            cv::imwrite("left_rectified.png", leftRect);
            cv::imwrite("right_rectified.png", rightRect);
            std::cout << "✓ Rectified images saved as left_rectified.png and right_rectified.png" << std::endl;

            return true;
        }
        else
        {
            std::cout << "✗ Rectification failed" << std::endl;
            return false;
        }
    }

    //? Function to test Disparity Mapping
    bool testDisparityMapping(const cv::Mat &leftRect, const cv::Mat &rightRect,
                              cv::Mat &disparityBM, cv::Mat &disparitySBGM)
    {
        std::cout << "\n=== Testing Disparity Mapping ===" << std::endl;

        // Test Block Matching
        std::cout << "\n--- Block Matching Algorithm ---" << std::endl;
        auto startBM = std::chrono::high_resolution_clock::now();

        disparityBM = stereoSystem->computeDisparityBM(leftRect, rightRect);

        auto endBM = std::chrono::high_resolution_clock::now();
        auto durationBM = std::chrono::duration_cast<std::chrono::milliseconds>(endBM - startBM);

        if (!disparityBM.empty())
        {
            std::cout << "✓ Block Matching disparity computed successfully" << std::endl;
            std::cout << "Time taken: " << durationBM.count() << " ms" << std::endl;
            std::cout << "Disparity size: " << disparityBM.size() << std::endl;
            std::cout << "Disparity type: " << disparityBM.type() << std::endl;

            // Normalize and save disparity map
            cv::Mat disparityBM_norm;
            cv::normalize(disparityBM, disparityBM_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::imwrite("disparity_BM.png", disparityBM_norm);
            std::cout << "✓ BM disparity map saved as disparity_BM.png" << std::endl;
        }
        else
        {
            std::cout << "✗ Block Matching disparity computation failed" << std::endl;
        }

        // Test Semi-Global Block Matching
        std::cout << "\n--- Semi-Global Block Matching Algorithm ---" << std::endl;
        auto startSBGM = std::chrono::high_resolution_clock::now();

        disparitySBGM = stereoSystem->computeDisparitySBGM(leftRect, rightRect);

        auto endSBGM = std::chrono::high_resolution_clock::now();
        auto durationSBGM = std::chrono::duration_cast<std::chrono::milliseconds>(endSBGM - startSBGM);

        if (!disparitySBGM.empty())
        {
            std::cout << "✓ SBGM disparity computed successfully" << std::endl;
            std::cout << "Time taken: " << durationSBGM.count() << " ms" << std::endl;
            std::cout << "Disparity size: " << disparitySBGM.size() << std::endl;
            std::cout << "Disparity type: " << disparitySBGM.type() << std::endl;

            // Normalize and save disparity map
            cv::Mat disparitySBGM_norm;
            cv::normalize(disparitySBGM, disparitySBGM_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::imwrite("disparity_SBGM.png", disparitySBGM_norm);
            std::cout << "✓ SBGM disparity map saved as disparity_SBGM.png" << std::endl;
        }
        else
        {
            std::cout << "✗ SBGM disparity computation failed" << std::endl;
        }

        return (!disparityBM.empty() || !disparitySBGM.empty());
    }

    bool testDepthMapping(const cv::Mat &disparity, const std::string &method)
    {
        std::cout << "\n=== Testing Depth Mapping (" << method << ") ===" << std::endl;

        if (disparity.empty())
        {
            std::cout << "✗ Input disparity map is empty" << std::endl;
            return false;
        }

        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat depthMap = stereoSystem->computeDepthMap(disparity);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        if (!depthMap.empty())
        {
            std::cout << "✓ Depth map computed successfully" << std::endl;
            std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
            std::cout << "Depth map size: " << depthMap.size() << std::endl;
            std::cout << "Depth map type: " << depthMap.type() << std::endl;

            // Analyze depth statistics
            double minDepth, maxDepth;
            cv::minMaxLoc(depthMap, &minDepth, &maxDepth);
            cv::Scalar meanDepth = cv::mean(depthMap);

            std::cout << "Depth statistics:" << std::endl;
            std::cout << "  Min depth: " << minDepth << " units" << std::endl;
            std::cout << "  Max depth: " << maxDepth << " units" << std::endl;
            std::cout << "  Mean depth: " << meanDepth[0] << " units" << std::endl;

            // Normalize and save depth map
            cv::Mat depthMap_norm;
            cv::normalize(depthMap, depthMap_norm, 0, 255, cv::NORM_MINMAX, CV_8U);

            // Apply colormap for better visualization
            cv::Mat depthMap_color;
            cv::applyColorMap(depthMap_norm, depthMap_color, cv::COLORMAP_JET);

            std::string filename = "depth_map_" + method + ".png";
            std::string colorFilename = "depth_map_" + method + "_color.png";

            cv::imwrite(filename, depthMap_norm);
            cv::imwrite(colorFilename, depthMap_color);

            std::cout << "✓ Depth map saved as " << filename << std::endl;
            std::cout << "✓ Colored depth map saved as " << colorFilename << std::endl;

            return true;
        }
        else
        {
            std::cout << "✗ Depth map computation failed" << std::endl;
            return false;
        }
    }

    //? 3d point reprojection
    void testPointReprojection(const cv::Mat &leftRect, const cv::Mat &disparity)
    {
        std::cout << "\n=== Testing 3D Point Reprojection ===" << std::endl;

        if (disparity.empty())
        {
            std::cout << "✗ Disparity map is empty, skipping reprojection test" << std::endl;
            return;
        }

        // Test a few points
        std::vector<cv::Point2f> testPoints = {
            cv::Point2f(leftRect.cols / 4, leftRect.rows / 4),         // Top-left quadrant
            cv::Point2f(leftRect.cols / 2, leftRect.rows / 2),         // Center
            cv::Point2f(3 * leftRect.cols / 4, 3 * leftRect.rows / 4), // Bottom-right quadrant
            cv::Point2f(leftRect.cols / 2, leftRect.rows / 4),         // Top-center
            cv::Point2f(leftRect.cols / 2, 3 * leftRect.rows / 4)      // Bottom-center
        };

        std::cout << "Testing 3D reprojection for sample points:" << std::endl;

        for (size_t i = 0; i < testPoints.size(); i++)
        {
            cv::Point2f point = testPoints[i];

            // Get disparity value at this point
            if (point.x >= 0 && point.x < disparity.cols && point.y >= 0 && point.y < disparity.rows)
            {
                float disparityValue = disparity.at<float>((int)point.y, (int)point.x);

                if (disparityValue > 0)
                {
                    cv::Point3f point3D = stereoSystem->reprojectTo3D(point, disparityValue);

                    std::cout << "Point " << (i + 1) << ": (" << point.x << ", " << point.y << ")" << std::endl;
                    std::cout << "  Disparity: " << disparityValue << std::endl;
                    std::cout << "  3D Point: (" << point3D.x << ", " << point3D.y << ", " << point3D.z << ")" << std::endl;
                }
                else
                {
                    std::cout << "Point " << (i + 1) << ": (" << point.x << ", " << point.y << ") - No valid disparity" << std::endl;
                }
            }
        }
    }

    //? Comparism visualisation
    void createComparison(const cv::Mat &leftImage, const cv::Mat &disparityBM, const cv::Mat &disparitySBGM)
    {
        std::cout << "\n=== Creating Comparison Visualization ===" << std::endl;

        if (leftImage.empty())
        {
            std::cout << "✗ Left image is empty" << std::endl;
            return;
        }

        // Create a comparison image
        cv::Mat comparison;
        cv::Mat leftResized, disparityBM_vis, disparitySBGM_vis;

        // Resize left image if needed
        cv::resize(leftImage, leftResized, cv::Size(640, 480));

        // Prepare disparity visualizations
        if (!disparityBM.empty())
        {
            cv::Mat disparityBM_norm;
            cv::normalize(disparityBM, disparityBM_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::resize(disparityBM_norm, disparityBM_vis, cv::Size(640, 480));
            cv::applyColorMap(disparityBM_vis, disparityBM_vis, cv::COLORMAP_JET);
        }
        else
        {
            disparityBM_vis = cv::Mat::zeros(480, 640, CV_8UC3);
            cv::putText(disparityBM_vis, "BM Failed", cv::Point(200, 240), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        }

        if (!disparitySBGM.empty())
        {
            cv::Mat disparitySBGM_norm;
            cv::normalize(disparitySBGM, disparitySBGM_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::resize(disparitySBGM_norm, disparitySBGM_vis, cv::Size(640, 480));
            cv::applyColorMap(disparitySBGM_vis, disparitySBGM_vis, cv::COLORMAP_JET);
        }
        else
        {
            disparitySBGM_vis = cv::Mat::zeros(480, 640, CV_8UC3);
            cv::putText(disparitySBGM_vis, "SBGM Failed", cv::Point(200, 240), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        }

        // Convert left image to color if it's grayscale
        if (leftResized.channels() == 1)
        {
            cv::cvtColor(leftResized, leftResized, cv::COLOR_GRAY2BGR);
        }

        // Create horizontal concatenation
        cv::Mat topRow, bottomRow;
        cv::hconcat(leftResized, disparityBM_vis, topRow);
        cv::hconcat(cv::Mat::zeros(480, 640, CV_8UC3), disparitySBGM_vis, bottomRow);
        cv::vconcat(topRow, bottomRow, comparison);

        // Add labels
        cv::putText(comparison, "Original Left", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        cv::putText(comparison, "BM Disparity", cv::Point(650, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        cv::putText(comparison, "SBGM Disparity", cv::Point(650, 510), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

        cv::imwrite("depth_mapping_comparison.png", comparison);
        std::cout << "✓ Comparison image saved as depth_mapping_comparison.png" << std::endl;
    }
};

int main()
{
    std::cout << "=== Stereo Depth Mapping Tester ===" << std::endl;

    // Configuration
    std::string calibrationFile = "stereo_calibration.xml";
    std::string leftImagePath = "../testingImages/testPair2/left camera view.png";
    std::string rightImagePath = "../testingImages/testPair2/right camera view.png";

    // Create tester
    DepthMappingTester tester(calibrationFile);

    // Step 1: Load calibration
    if (!tester.loadCalibration())
    {
        std::cout << "\n✗ Cannot proceed without valid calibration. Exiting." << std::endl;
        return -1;
    }

    // Step 2: Load test images
    cv::Mat leftImage, rightImage;
    if (!tester.loadTestImages(leftImagePath, rightImagePath, leftImage, rightImage))
    {
        std::cout << "\n✗ Cannot proceed without valid test images. Exiting." << std::endl;
        return -1;
    }

    // Step 3: Test rectification
    cv::Mat leftRect, rightRect;
    if (!tester.testRectification(leftImage, rightImage, leftRect, rightRect))
    {
        std::cout << "\n✗ Rectification failed. Cannot proceed to depth mapping." << std::endl;
        return -1;
    }

    // Add this debug info before calling computeDisparityBM
    std::cout << "Debug info:" << std::endl;
    std::cout << "Left rectified channels: " << leftRect.channels() << std::endl;
    std::cout << "Right rectified channels: " << rightRect.channels() << std::endl;
    std::cout << "Left rectified type: " << leftRect.type() << std::endl;
    std::cout << "Right rectified type: " << rightRect.type() << std::endl;

    // Step 4: Test disparity mapping
    cv::Mat disparityBM, disparitySBGM;
    if (!tester.testDisparityMapping(leftRect, rightRect, disparityBM, disparitySBGM))
    {
        std::cout << "\n✗ All disparity computation methods failed." << std::endl;
        return -1;
    }

    // Step 5: Test depth mapping
    bool depthTestSuccess = false;

    if (!disparityBM.empty())
    {
        if (tester.testDepthMapping(disparityBM, "BM"))
        {
            depthTestSuccess = true;
            tester.testPointReprojection(leftRect, disparityBM);
        }
    }

    if (!disparitySBGM.empty())
    {
        if (tester.testDepthMapping(disparitySBGM, "SBGM"))
        {
            depthTestSuccess = true;
            tester.testPointReprojection(leftRect, disparitySBGM);
        }
    }

    // Step 6: Create comparison visualization
    tester.createComparison(leftImage, disparityBM, disparitySBGM);

    // Final summary
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "✓ Calibration loading: SUCCESS" << std::endl;
    std::cout << "✓ Image loading: SUCCESS" << std::endl;
    std::cout << "✓ Rectification: SUCCESS" << std::endl;
    std::cout << "✓ Disparity mapping: " << ((!disparityBM.empty() || !disparitySBGM.empty()) ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "✓ Depth mapping: " << (depthTestSuccess ? "SUCCESS" : "FAILED") << std::endl;

    std::cout << "\nGenerated files:" << std::endl;
    std::cout << "- left_rectified.png" << std::endl;
    std::cout << "- right_rectified.png" << std::endl;
    std::cout << "- disparity_BM.png (if BM succeeded)" << std::endl;
    std::cout << "- disparity_SBGM.png (if SBGM succeeded)" << std::endl;
    std::cout << "- depth_map_*.png (if depth mapping succeeded)" << std::endl;
    std::cout << "- depth_mapping_comparison.png" << std::endl;

    std::cout << "\n✓ Depth mapping test completed!" << std::endl;

    return 0;
}
