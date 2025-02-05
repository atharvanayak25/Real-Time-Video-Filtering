//  ATHARVA NAYAK  //
// NU ID: 002322653  //

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int colorfulFace(cv::Mat &src, cv::Mat &dst, cv::CascadeClassifier &faceCascade);
int blurOutsideFace(cv::Mat &src, cv::Mat &dst, cv::CascadeClassifier &faceCascade);
int customGreyscale(cv::Mat &src, cv::Mat &dst);
int applySepiaFilter(cv::Mat &src, cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);
int embossEffect(cv::Mat &src, cv::Mat &dst);
void addCaptions(cv::Mat &img, const std::string &topText, const std::string &bottomText);
void applyFrameMask(cv::Mat &frame, const cv::Mat &mask, cv::Rect visibleRegion, double alpha = 1);



#endif 
