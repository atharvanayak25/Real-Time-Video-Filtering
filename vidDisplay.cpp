//  ATHARVA NAYAK  //
// NU ID: 002322653  //

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include "faceDetect.h"
#include "filter.h"

// namespace fs = std::filesystem;

// Function prototypes
// int blur5x5_2(cv::Mat &src, cv::Mat &dst);
// int colorfulFace(cv::Mat &src, cv::Mat &dst, cv::CascadeClassifier &faceCascade);
// int blurOutsideFace(cv::Mat &src, cv::Mat &dst, cv::CascadeClassifier &faceCascade);
// int customGreyscale(cv::Mat &src, cv::Mat &dst);
// int applySepiaFilter(cv::Mat &src, cv::Mat &dst);
// int sobelX3x3(cv::Mat &src, cv::Mat &dst);
// int sobelY3x3(cv::Mat &src, cv::Mat &dst);
// int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
// int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);
// int embossEffect(cv::Mat &src, cv::Mat &dst);
// void addCaptions(cv::Mat &img, const std::string &topText, const std::string &bottomText);
// void applyFrameMask(cv::Mat &frame, const cv::Mat &mask, cv::Rect visibleRegion, double alpha = 1);

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video device." << std::endl;
        return -1;
    }

    cv::CascadeClassifier faceCascade;
    if (!faceCascade.load("C:/Users/Atharva/NEU_DOCS/Proj1/Proj_1_Main/haarcascade_frontalface_alt2.xml")) {
        std::cerr << "Error: Unable to load face cascade." << std::endl;
        return -1;
    }

    // Load the mask image for addFrame feature
    cv::Mat mask = cv::imread("C:/Users/Atharva/NEU_DOCS/Proj1/Proj_1_Main/frame1.png");
    if (mask.empty()) {
        std::cerr << "Error: Unable to load mask image." << std::endl;
        return -1;
    }

    // Matrices for different filters
    cv::Mat frame, filteredFrame, grayFrame, customGrayFrame, sepiaFrame;
    cv::Mat blurredFrame, sobelXFrame, sobelYFrame, gradientMagnitudeFrame;
    cv::Mat quantizedFrame, embossedFrame, blurredOutsideFaceFrame, colorfulFaceFrame;

    char mode = 'c'; // Default mode: color
    int saveCount = 0;

    // Parameters for visible rectangular region for the mask
    cv::Rect visibleRegion(75, 75, 480, 320); // x, y, width, height

    // Create original and filtered video window
    cv::namedWindow("Original Video", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Filtered Video", cv::WINDOW_AUTOSIZE);

    std::cout << "Controls:\n"
              << "  'f' - Toggle face detection\n"
              << "  'o' - Blur outside face\n"
              << "  'e' - Emboss effect\n"
              << "  'z' - Colorful face with grayscale background\n"
              << "  'b' - Blur mode\n"
              << "  'g' - Grayscale mode\n"
              << "  'h' - Custom grayscale mode\n"
              << "  'v' - Sepia mode\n"
              << "  'x' - Sobel X\n"
              << "  'y' - Sobel Y\n"
              << "  'm' - Gradient magnitude\n"
              << "  'l' - Blur quantization\n"
              << "  'c' - Default color mode\n"
              << "  'r' - Add Captions\n"
              << "  'q' - Quit\n";

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Empty frame captured." << std::endl;
            break;
        }

        // Set default to original frame
        filteredFrame = frame.clone();

        // Key mode handling with if-else statements
        if (mode == 'f') {
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
            std::vector<cv::Rect> faces;
            detectFaces(grayFrame, faces);
            drawBoxes(filteredFrame, faces);
        } else if (mode == 'o') {
            blurOutsideFace(frame, blurredOutsideFaceFrame, faceCascade);
            filteredFrame = blurredOutsideFaceFrame;
        } else if (mode == 'e') {
            embossEffect(frame, embossedFrame);
            filteredFrame = embossedFrame;
        } else if (mode == 'z') {
            colorfulFace(frame, colorfulFaceFrame, faceCascade);
            filteredFrame = colorfulFaceFrame;
        } else if (mode == 'g') {
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
            cv::cvtColor(grayFrame, filteredFrame, cv::COLOR_GRAY2BGR);
        } else if (mode == 'h') {
            customGreyscale(frame, customGrayFrame);
            filteredFrame = customGrayFrame;
        } else if (mode == 'v') {
            applySepiaFilter(frame, sepiaFrame);
            filteredFrame = sepiaFrame;
        } else if (mode == 'b') {
            blur5x5_2(frame, blurredFrame);
            filteredFrame = blurredFrame;
        } else if (mode == 'x') {
            sobelX3x3(frame, sobelXFrame);
            cv::convertScaleAbs(sobelXFrame, filteredFrame);
        } else if (mode == 'y') {
            sobelY3x3(frame, sobelYFrame);
            cv::convertScaleAbs(sobelYFrame, filteredFrame);
        } else if (mode == 'm') {
            sobelX3x3(frame, sobelXFrame);
            sobelY3x3(frame, sobelYFrame);
            magnitude(sobelXFrame, sobelYFrame, gradientMagnitudeFrame);
            filteredFrame = gradientMagnitudeFrame;
        } else if (mode == 'l') {
            blurQuantize(frame, quantizedFrame, 10);
            filteredFrame = quantizedFrame;
        } else if (mode == 'p') {
            applyFrameMask(filteredFrame, mask, visibleRegion);
        } else if (mode == 'r') {
            mode = 'c';

            std::string topText, bottomText;

            std::cout << "Enter Top Caption: ";
            std::getline(std::cin, topText);
            std::cout << "Enter Bottom Caption: ";
            std::getline(std::cin, bottomText);

            cv::Mat memeFrame = frame.clone();
            addCaptions(memeFrame, topText, bottomText);

            // Show the meme
            cv::imshow("Meme Preview", memeFrame);
            cv::waitKey(1);

            // Save the meme automatically
            std::string savePath = "C:/Users/Atharva/NEU_DOCS/Proj1/Proj_1_Main/images/Meme/meme_with_captions.jpg"; // Define save path and filename
            if (cv::imwrite(savePath, memeFrame)) {
                std::cout << "Meme saved successfully at: " << savePath << std::endl;
            } else {
                 std::cerr << "Error: Could not save the meme." << std::endl;
            }
        }

        // Display original and filtered video
        cv::imshow("Original Video", frame);
        cv::imshow("Filtered Video", filteredFrame);

        // Handle key press events
        char key = cv::waitKey(10);
        if (key == 'q') {
            std::cout << "Exiting program..." << std::endl;
            break;
        } else if (key == 's') {
            // Save the current filtered frame
            static int saveCount = 0;

            std::string baseDir = "C:/Users/Atharva/NEU_DOCS/Proj1/Proj_1_Main/images/";
            std::string modeDir;

            if (mode == 'f') modeDir = "Face_Detection/";
            else if (mode == 'o') modeDir = "Blur_Outside_Face/";
            else if (mode == 'e') modeDir = "Emboss_Effect/";
            else if (mode == 'z') modeDir = "Colorful_Face/";
            else if (mode == 'g') modeDir = "Grayscale/";
            else if (mode == 'h') modeDir = "Custom_Grayscale/";
            else if (mode == 'v') modeDir = "Sepia/";
            else if (mode == 'b') modeDir = "Blur/";
            else if (mode == 'x') modeDir = "SobelX/";
            else if (mode == 'y') modeDir = "SobelY/";
            else if (mode == 'm') modeDir = "Gradient_Magnitude/";
            else if (mode == 'l') modeDir = "Blur_Quantization/";
            else if (mode == 'r') modeDir = "Meme/";
            else if (mode == 'p') modeDir = "Photo_Frame/";
            else modeDir = "Default_Color/";

            std::string fullDir = baseDir + modeDir;

            std::string filename = fullDir + "filtered_frame_" + std::to_string(saveCount++) + ".jpg";

            if (!filteredFrame.empty()) {
                cv::imwrite(filename, filteredFrame);
                std::cout << "Filtered frame saved as: " << filename << std::endl;
            } else {
                std::cerr << "Error: No frame available to save." << std::endl;
            }
        } else if (key >= 'a' && key <= 'z') {
            mode = key;
            std::cout << "Mode switched to: " << mode << std::endl;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
