//  ATHARVA NAYAK  //
// NU ID: 002322653  //


#include <opencv2/opencv.hpp>
#include <iostream>


int customGreyscale(cv::Mat &src, cv::Mat &dst) {
    // Check if the input image is valid
    if (src.empty()) {
        return -1; // Return error if the source image is empty
    }

    // Ensure the destination has the same size and type as the source
    dst.create(src.size(), src.type());

    // Iterate through each pixel to apply the custom grayscale transformation
    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            // Access the pixel in the source image
            cv::Vec3b intensity = src.at<cv::Vec3b>(row, col);
            uchar blue = intensity[0];
            uchar green = intensity[1];
            uchar red = intensity[2];

            // Custom grayscale transformation: invert the red channel and average it with green and blue
            uchar customGray = static_cast<uchar>((255 - red + green + blue) / 3);

            // Set all color channels in the destination image to the computed grayscale value
            dst.at<cv::Vec3b>(row, col) = cv::Vec3b(customGray, customGray, customGray);
        }
    }

    return 0;
}
// Sepia filter function
int applySepiaFilter(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        std::cerr << "Source image is empty" << std::endl;
        return -1;
    }

    // Ensure the destination image has the same size and type as the source
    dst = cv::Mat(src.size(), src.type());

    // Apply the sepia tone transformation
    for (int row = 0; row < src.rows; ++row) {
        for (int col = 0; col < src.cols; ++col) {
            // Access the BGR pixel values
            cv::Vec3b pixel = src.at<cv::Vec3b>(row, col);

            // Extract original B, G, R values
            uchar originalBlue = pixel[0];
            uchar originalGreen = pixel[1];
            uchar originalRed = pixel[2];

            // Calculate new values using the sepia transformation matrix
            int newBlue = static_cast<int>(0.272 * originalRed + 0.534 * originalGreen + 0.131 * originalBlue);
            int newGreen = static_cast<int>(0.349 * originalRed + 0.686 * originalGreen + 0.168 * originalBlue);
            int newRed = static_cast<int>(0.393 * originalRed + 0.769 * originalGreen + 0.189 * originalBlue);

            // the values to ensure they are within [0, 255]
            newBlue = std::min(255, newBlue);
            newGreen = std::min(255, newGreen);
            newRed = std::min(255, newRed);

            // Assign the new values to the destination image
            dst.at<cv::Vec3b>(row, col) = cv::Vec3b(newBlue, newGreen, newRed);
        }
    }

    return 0;
}


int blur5x5_1(cv::Mat &src, cv::Mat &dst) {
    // Ensure the input image is of type CV_8UC3
    if (src.empty() || src.type() != CV_8UC3) {
        std::cerr << "Input image must be a non-empty 8-bit 3-channel image." << std::endl;
        return -1;
    }


    dst = src.clone();

    // Gaussian kernel for 5x5 blur
    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };

    int kernel_sum = 256; // Sum of the kernel values

    // Loop through the image excluding the outer two rows and columns
    for (int y = 2; y < src.rows - 2; ++y) {
        for (int x = 2; x < src.cols - 2; ++x) {
            // Separate accumulators for each color channel
            int sumB = 0, sumG = 0, sumR = 0;

            // Apply the kernel to the pixel and its neighbors
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(y + ky, x + kx);
                    int weight = kernel[ky + 2][kx + 2];

                    sumB += pixel[0] * weight;
                    sumG += pixel[1] * weight;
                    sumR += pixel[2] * weight;
                }
            }

            // Normalize and clamp the pixel values to [0, 255]
            dst.at<cv::Vec3b>(y, x)[0] = std::min(255, std::max(0, sumB / kernel_sum));
            dst.at<cv::Vec3b>(y, x)[1] = std::min(255, std::max(0, sumG / kernel_sum));
            dst.at<cv::Vec3b>(y, x)[2] = std::min(255, std::max(0, sumR / kernel_sum));
        }
    }

    return 0;
}


int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    // Check if the input image is valid
    if (src.empty() || src.channels() != 3) {
        return -1; // Return error if the source image is invalid
    }

    dst = src.clone();
    cv::Mat temp = src.clone();

    // Kernel weights for Gaussian blur
    int kernel[5] = {1, 2, 4, 2, 1};
    int kernelSum = 16; // Sum of kernel weights

    // Step 1: Horizontal pass
    for (int y = 0; y < src.rows; y++) {
        for (int x = 2; x < src.cols - 2; x++) {
            int blueSum = 0, greenSum = 0, redSum = 0;

            for (int k = -2; k <= 2; k++) {
                int weight = kernel[k + 2];
                cv::Vec3b pixel = src.at<cv::Vec3b>(y, x + k);

                blueSum += pixel[0] * weight;
                greenSum += pixel[1] * weight;
                redSum += pixel[2] * weight;
            }

            temp.at<cv::Vec3b>(y, x)[0] = blueSum / kernelSum;
            temp.at<cv::Vec3b>(y, x)[1] = greenSum / kernelSum;
            temp.at<cv::Vec3b>(y, x)[2] = redSum / kernelSum;
        }
    }

    // Step 2: Vertical pass
    for (int y = 2; y < src.rows - 2; y++) {
        for (int x = 0; x < src.cols; x++) {
            int blueSum = 0, greenSum = 0, redSum = 0;

            for (int k = -2; k <= 2; k++) {
                int weight = kernel[k + 2];
                cv::Vec3b pixel = temp.at<cv::Vec3b>(y + k, x);

                blueSum += pixel[0] * weight;
                greenSum += pixel[1] * weight;
                redSum += pixel[2] * weight;
            }

            dst.at<cv::Vec3b>(y, x)[0] = blueSum / kernelSum;
            dst.at<cv::Vec3b>(y, x)[1] = greenSum / kernelSum;
            dst.at<cv::Vec3b>(y, x)[2] = redSum / kernelSum;
        }
    }

    return 0;
}

int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.type() != CV_8UC3) {
        std::cerr << "Source image is empty" << std::endl;
        return -1;
    }

    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    // Sobel kernel for X direction
    int kernelX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    for (int y = 1; y < src.rows - 1; ++y) {
        for (int x = 1; x < src.cols - 1; ++x) {
            for (int c = 0; c < 3; ++c) { // Process each channel
                int sum = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        sum += kernelX[ky + 1][kx + 1] * src.at<cv::Vec3b>(y + ky, x + kx)[c];
                    }
                }
                dst.at<cv::Vec3s>(y, x)[c] = sum;
            }
        }
    }

    return 0;
}

int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.type() != CV_8UC3) {
        std::cerr << "Source image is empty" << std::endl;
        return -1;
    }

    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    // Sobel kernel for Y direction
    int kernelY[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (int y = 1; y < src.rows - 1; ++y) {
        for (int x = 1; x < src.cols - 1; ++x) {
            for (int c = 0; c < 3; ++c) { // Process each channel
                int sum = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        sum += kernelY[ky + 1][kx + 1] * src.at<cv::Vec3b>(y + ky, x + kx)[c];
                    }
                }
                dst.at<cv::Vec3s>(y, x)[c] = sum;
            }
        }
    }

    return 0;
}


int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    if (sx.empty() || sy.empty() || sx.size() != sy.size() || sx.type() != CV_16SC3 || sy.type() != CV_16SC3) {
        std::cerr << "Invalid input images for magnitude calculation." << std::endl;
        return -1;
    }

    // Create the output image of type CV_8UC3 for display
    dst = cv::Mat::zeros(sx.size(), CV_8UC3);

    for (int y = 0; y < sx.rows; ++y) {
        for (int x = 0; x < sx.cols; ++x) {
            // Access the gradient values for each channel
            cv::Vec3s gradX = sx.at<cv::Vec3s>(y, x);
            cv::Vec3s gradY = sy.at<cv::Vec3s>(y, x);
            cv::Vec3b &outputPixel = dst.at<cv::Vec3b>(y, x);

            for (int c = 0; c < 3; ++c) {
                // Calculate the magnitude using Euclidean distance
                float mag = std::sqrt(gradX[c] * gradX[c] + gradY[c] * gradY[c]);
                // the value to the range [0, 255] for display
                outputPixel[c] = static_cast<uchar>(std::min(255.0f, mag));
            }
        }
    }

    return 0;
}

int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    if (src.empty() || src.type() != CV_8UC3) {
        std::cerr << "Error: Invalid input image." << std::endl;
        return -1;
    }

    // Step 1: Blur the image
    cv::Mat blurred;
    if (blur5x5_2(src, blurred) != 0) {
        std::cerr << "Error: Blur operation failed in blurQuantize." << std::endl;
        return -1;
    }

    // Step 2: Quantize the image
    int bucketSize = 255 / levels;
    dst = cv::Mat::zeros(blurred.size(), blurred.type());
    for (int y = 0; y < blurred.rows; ++y) {
        for (int x = 0; x < blurred.cols; ++x) {
            cv::Vec3b pixel = blurred.at<cv::Vec3b>(y, x);
            for (int c = 0; c < 3; ++c) {
                int quantizedValue = (pixel[c] / bucketSize) * bucketSize;
                dst.at<cv::Vec3b>(y, x)[c] = static_cast<uchar>(quantizedValue);
            }
        }
    }

    return 0;

}

// Emboss effect
int embossEffect(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        std::cerr << "Source image is empty" << std::endl;
        return -1;
    }

    cv::Mat sobelX, sobelY;
    if (sobelX3x3(src, sobelX) != 0 || sobelY3x3(src, sobelY) != 0) {
        std::cerr << "Error in Sobel operation for emboss effect" << std::endl;
        return -1;
    }

    dst = cv::Mat::zeros(src.size(), CV_8UC3);

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            cv::Vec3s gx = sobelX.at<cv::Vec3s>(y, x);
            cv::Vec3s gy = sobelY.at<cv::Vec3s>(y, x);

            for (int c = 0; c < 3; ++c) {
                float dot = gx[c] * 0.7071f + gy[c] * 0.7071f;
                int intensity = std::max(0, std::min(255, static_cast<int>(dot) + 128));
                dst.at<cv::Vec3b>(y, x)[c] = static_cast<uchar>(intensity);
            }
        }
    }

    return 0;
}

// Colorful face with grayscale background
int colorfulFace(cv::Mat &src, cv::Mat &dst, cv::CascadeClassifier &faceCascade) {
    if (src.empty()) {
        std::cerr << "Source image is empty" << std::endl;
        return -1;
    }

    dst = cv::Mat::zeros(src.size(), src.type());
    cv::Mat grayFrame;
    cv::cvtColor(src, grayFrame, cv::COLOR_BGR2GRAY);
    cv::cvtColor(grayFrame, dst, cv::COLOR_GRAY2BGR);

    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(src, faces, 1.1, 3, 0, cv::Size(50, 50));

    for (const auto &face : faces) {
        src(face).copyTo(dst(face));
    }

    return 0;
}

// Blur Outside Face
int blurOutsideFace(cv::Mat &src, cv::Mat &dst, cv::CascadeClassifier &faceCascade) {
    if (src.empty()) {
        std::cerr << "Source image is empty" << std::endl;
        return -1;
    }

    // Detect faces
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(src, faces, 1.1, 3, 0, cv::Size(50, 50));

    // Create a blurred version of the input image
    cv::Mat blurred;
    if (blur5x5_2(src, blurred) != 0) {
        std::cerr << "Error: Blur operation failed in blurOutsideFace." << std::endl;
        return -1;
    }

    // Start with the blurred image
    dst = blurred.clone();

    // Copy the face regions from the original image to the destination
    for (const auto &face : faces) {
        src(face).copyTo(dst(face));
    }

    return 0;
}
