//  ATHARVA NAYAK  //
// NU ID: 002322653  //

#include <opencv2/opencv.hpp>
#include <string>

// Function to add captions to an image
void addCaptions(cv::Mat& img, const std::string& topText, const std::string& bottomText) {
    // Font and text settings
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.0;
    int thickness = 2;
    int baseline = 0;

    // Add top caption
    cv::Size topTextSize = cv::getTextSize(topText, fontFace, fontScale, thickness, &baseline);
    cv::Point topTextOrg((img.cols - topTextSize.width) / 2, topTextSize.height + 10);
    cv::putText(img, topText, topTextOrg, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);

    // Add bottom caption
    cv::Size bottomTextSize = cv::getTextSize(bottomText, fontFace, fontScale, thickness, &baseline);
    cv::Point bottomTextOrg((img.cols - bottomTextSize.width) / 2, img.rows - 10);
    cv::putText(img, bottomText, bottomTextOrg, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
}
