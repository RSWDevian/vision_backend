#include "../include/stereo.hpp"
#include <iostream>
#include <fstream>

//? Constructor matrix of the stereo vision class
StereoVision::StereoVision() : isCalibrated(false) {
    // Initialize matrices to empty
    cameraMatrix1 = cv::Mat();
    cameraMatrix2 = cv::Mat();
    distCoeffs1 = cv::Mat();
    distCoeffs2 = cv::Mat();
    R = cv::Mat();
    T = cv::Mat();
    E = cv::Mat();
    F = cv::Mat();
    R1 = cv::Mat();
    R2 = cv::Mat();
    P1 = cv::Mat();
    P2 = cv::Mat();
    Q = cv::Mat();
    map1x = cv::Mat();
    map1y = cv::Mat();
    map2x = cv::Mat();
    map2y = cv::Mat();
}

//? Destructor matrix of the stereo vision class
StereoVision::~StereoVision() {
    // Release all matrices
    cameraMatrix1.release();
    cameraMatrix2.release();
    distCoeffs1.release();
    distCoeffs2.release();
    R.release();
    T.release();
    E.release();
    F.release();
    R1.release();
    R2.release();
    P1.release();
    P2.release();
    Q.release();
    map1x.release();
    map1y.release();
    map2x.release();
    map2y.release();
}

//? Calibrate function of StereoVision class
bool StereoVision::calibrateStereo(
    const std::vector<std::vector<cv::Point3f>>& objectPoints,
    const std::vector<std::vector<cv::Point2f>>& imagePoints1,
    const std::vector<std::vector<cv::Point2f>>& imagePoints2,
    const cv::Size& imgSize 
){
    // checking conditions
    if(objectPoints.empty() || imagePoints1.empty() || imagePoints2.empty()){
        std::cerr<<"Error: Not enough points for calibration"<<std::endl;
        return false;
    }

    if(objectPoints.size() != imagePoints1.size() || objectPoints.size() != imagePoints2.size()){
        std::cerr<<"Error: Mismatched in object point and image point sets"<<std::endl;
        return false;
    }

    // storing the image size
    imageSize = imgSize;
    // Indivual camera calibration
    std::vector<cv::Mat> rvecs1, tvecs1, rvecs2, tvecs2;

    double rms1 = cv::calibrateCamera(objectPoints, imagePoints1, imgSize, cameraMatrix1, distCoeffs1, rvecs1, tvecs1);
    double rms2 = cv::calibrateCamera(objectPoints, imagePoints2, imgSize, cameraMatrix2, distCoeffs2, rvecs2, tvecs2);
    // Output of individual camera calibration
    std::cout << "\n=== Individual Calibration ===" << std::endl;
    std::cout << "Camera 1 RMS error = " << rms1 << std::endl;
    std::cout << "Camera 2 RMS error = " << rms2 << std::endl;
    // Stereo calibration
    double rms = cv::stereoCalibrate(objectPoints, imagePoints1, imagePoints2,
                                    cameraMatrix1, distCoeffs1,
                                    cameraMatrix2, distCoeffs2,
                                    imageSize, R, T, E, F,
                                    cv::CALIB_FIX_INTRINSIC);
    
    
    std::cout << "\n=== Stereo Calibration ===" << std::endl;
    std::cout << "Stereo RMS error = " << rms << std::endl;

    // Matrices for debugging
    std::cout << "Camera Matrix 1:\n" << cameraMatrix1 << std::endl;
    std::cout << "Distortion Coeffs 1:\n" << distCoeffs1.t() << std::endl;

    std::cout << "Camera Matrix 2:\n" << cameraMatrix2 << std::endl;
    std::cout << "Distortion Coeffs 2:\n" << distCoeffs2.t() << std::endl;

    std::cout << "Rotation Matrix (R):\n" << R << std::endl;
    std::cout << "Translation Vector (T):\n" << T << std::endl;
    std::cout << "Essential Matrix (E):\n" << E << std::endl;
    std::cout << "Fundamental Matrix (F):\n" << F << std::endl;
    
    if(rms<1){
        std::cout<<"[Ok] Good stereo calibration"<<std::endl;
    }else{
        std::cout<<"[Warn] Poor stereo calibration, check input points!"<<std::endl;
    }
    isCalibrated = true;
    return true;
}

//? Load function to load the saved calibration matrixes
bool StereoVision::loadCalibration(const std::string& filename){
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if(!fs.isOpened()){
        std::cerr<<"Failed to open calibrated file:"<<filename<<std::endl;
        return false;
    }
    std::cout<<"Uploading the camera parameters..."<<std::endl;
    fs["cameraMatrix1"] >> cameraMatrix1;
    fs["cameraMatrix2"] >> cameraMatrix2;
    fs["distCoeffs1"] >> distCoeffs1;
    fs["distCoeffs2"] >> distCoeffs2;
    fs["R"] >> R;
    fs["T"] >> T;
    fs["E"] >> E;
    fs["F"] >> F;
    fs["imageSize"] >> imageSize;
    fs.release();
    isCalibrated = true;
    return true;
}

//? Save function to save the calculated calibrated data
bool StereoVision::saveCalibration(const std::string& filename){
    if(!isCalibrated) return false;

    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if(!fs.isOpened()) return false;
    std::cout<<"Saving the camera parameters..."<<std::endl;
    fs << "cameraMatrix1" << cameraMatrix1;
    fs << "cameraMatrix2" << cameraMatrix2;
    fs << "distCoeffs1" << distCoeffs1;
    fs << "distCoeffs2" << distCoeffs2;
    fs << "R" << R;
    fs << "T" << T;
    fs << "E" << E;
    fs << "F" << F;
    fs << "imageSize" << imageSize;
    fs.release();
    return true;
}

//? Function to rectify the images
bool StereoVision::rectifyImages(
    const cv::Mat& left, const cv::Mat& right,
    cv::Mat& leftRect, cv::Mat& rightRect
){

    if(!isCalibrated){
        std::cerr<<"Error: StereoVision is not calibrated"<<std::endl;
        return false;
    }

    // Rectify the images
    cv::stereoRectify(cameraMatrix1, distCoeffs1,
                      cameraMatrix2, distCoeffs2,
                      imageSize, R, T,
                      R1, R2, P1, P2, Q);
    // Comute rectification maps
    cv::initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, CV_32FC1, map1x, map1y);
    cv::initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, imageSize, CV_32FC1, map2x, map2y);
    
    // Apply rectification
    cv::remap(left, leftRect, map1x, map1y, cv::INTER_LINEAR);
    cv::remap(right, rightRect, map2x, map2y, cv::INTER_LINEAR);

    return true;
}

//? Compute disparity using Block Matching
cv::Mat StereoVision::computeDisparityBM(
    const cv::Mat& leftRect,
    const cv::Mat& rightRect
){
    cv::Mat disparity;
    cv::Ptr<cv::StereoBM> stereoBM = cv::StereoBM::create();
    stereoBM->compute(leftRect, rightRect, disparity);
    return disparity;
}

//? Compute disparity using Semi-Global Block Matching
cv::Mat StereoVision::computeDisparitySBGM(const cv::Mat& leftRect, const cv::Mat& rightRect) {
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 96, 9);
    cv::Mat disparity;
    sgbm->compute(leftRect, rightRect, disparity);
    return disparity;
}

//? Filter occlusions from disparity map
cv::Mat StereoVision::filterOcculsions(const cv::Mat& disparity) {
    cv::Mat filtered;
    disparity.copyTo(filtered);
    
    // Simple occlusion filtering - remove invalid disparities
    for (int y = 0; y < filtered.rows; y++) {
        for (int x = 0; x < filtered.cols; x++) {
            short d = filtered.at<short>(y, x);
            if (d <= 0) {
                filtered.at<short>(y, x) = 0;
            }
        }
    }
    
    // Apply median filter to reduce noise
    cv::medianBlur(filtered, filtered, 5);
    
    return filtered;
}

//? Reproject 2D point to 3D
cv::Point3f StereoVision::reprojectTo3D(const cv::Point2f& point, float disparity) {
    if (disparity <= 0) return cv::Point3f(0, 0, 0);
    
    float w = Q.at<double>(3, 2) * point.x + Q.at<double>(3, 3);
    cv::Point3f point3D;
    point3D.x = (Q.at<double>(0, 0) * point.x + Q.at<double>(0, 3)) / w;
    point3D.y = (Q.at<double>(1, 1) * point.y + Q.at<double>(1, 3)) / w;
    point3D.z = Q.at<double>(2, 3) / w;
    
    return point3D;
}