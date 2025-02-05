//  ATHARVA NAYAK  //
// NU ID: 002322653  //

#include <opencv2/opencv.hpp>
#include <iostream>

void applyFrameMask(cv::Mat &frame, const cv::Mat &mask, cv::Rect visibleRegion, double alpha = 1) {
    if (mask.empty()) {
        std::cerr << "Error: Mask image is empty!" << std::endl;
        return;
    }

    // Resize the mask to match the video frame dimensions
    cv::Mat resizedMask;
    cv::resize(mask, resizedMask, frame.size());

    // Ensure the mask has three channels for blending
    if (resizedMask.channels() == 4) {
        // Convert RGBA mask to BGR
        cv::cvtColor(resizedMask, resizedMask, cv::COLOR_BGRA2BGR);
    }

    // Create a copy of the original frame to overlay the visible region
    cv::Mat outputFrame = resizedMask.clone();

    // Copy the visible region (rectangular) from the original frame to the output frame
    frame(visibleRegion).copyTo(outputFrame(visibleRegion));

    // Blend the mask with the frame outside the visible region
    cv::addWeighted(outputFrame, alpha, resizedMask, 1.0 - alpha, 0, frame);
}
