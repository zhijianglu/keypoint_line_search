//
// Created by lab on 2020/8/24.
//

#ifndef KEYPOINTS_H
#define KEYPOINTS_H
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;
using namespace Eigen;

void extr_match_kpts_cv(
	Mat& img_1,
	Mat& img_2,
	vector<Point2i>& kpts_1,
	vector<Point2i>& kpts_2
);

void selectPt_and_match(
	double min_distance,
	std::vector<KeyPoint> &keypoints_1,vector<Point2i>& kpts_1, Mat &descriptors_1,
	std::vector<KeyPoint> &keypoints_2,vector<Point2i>& kpts_2, Mat &descriptors_2);

void extract_feature(Mat& input_mat,std::vector<KeyPoint>& keypoints,Mat& descriptors);


#endif //KEYPOINTS_H
