//
// Created by lab on 2020/8/23.
//

#ifndef TOOL_H
#define TOOL_H
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

using namespace cv;
using namespace std;
using namespace Eigen;


void depth_prj(double depth_factor, Matrix3d &K,
			   Matrix3d &Rrc, Vector3d &trc,
			   Mat &depth_ref, Mat &depth_curr, Mat &depth_curr_prj2_ref);


void drawKeyPoints_num(
	Scalar color,
	Mat & img,std::vector<Point2i>& keypoints
);

void drawKeyPoints_num(
	Scalar color,
	Mat & img,std::vector<Point2f>& keypoints
);

void calc_gradient(Mat &img, Mat &gradient_length,Mat &gradient_x,Mat &gradient_y);

void search_point(
	Matrix3d K,

	Matrix3d R_cr,
	Vector3d t_cr,

	const int &x, //当前坐标在keyframe图像上的像素位置
	const int &y,

	const double &max_idep,
	const double &min_idep,

	Vector2f &epipolar_line,
	Mat &keyframe_image_tex,
	Mat &currframe_image_tex,
	Mat &result_show,
	float& best_idepth,
	float& alpha
);

void search_point_v1(
	Matrix3d K,

	Matrix3d R_cr,
	Vector3d t_cr,

	const int &x, //当前坐标在keyframe图像上的像素位置
	const int &y,

	const double &max_idep,
	const double &min_idep,

	Vector2f& epipolar_line,
	Mat & keyframe_image_tex,
	Mat & currframe_image_tex,
	Mat & result_show,
	float& best_idepth,
	float& alpha

);

#endif //TOOL_H
