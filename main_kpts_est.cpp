#include "/home/lab/cpp_lib/getfile.h"
#include <iomanip>
#include <vector>
#include <iostream>
#include <string>

//my functions
#include "tool.h"
#include "keypoints.h"
using namespace cv;
using namespace std;

string path = "../data/test2";
#define MIN_GRAIDIENT 5
#define MIN_INV_DEPTH 0.33  // 2.0m
#define MAX_INV_DEPTH 2   // 0.5m

int main(){
//	读取相机参数
	double fx = 517.3;
	double fy = 516.5;
	double cx = 318.6;
	double cy = 255.3;
	Matrix3d K;
	Matrix3d K_inv;
	K.setZero();
	K(0,0) = fx;
	K(1,1) = fy;
	K(0,2) = cx;
	K(1,2) = cy;
	K(2,2) = 1;
	cout << "K : \n" << K << endl;
	K_inv = K.inverse();

	double width = 640;
	double height= 480;

//  读取以及转换位姿
	string pose_file_path = path+"/gt.txt";
	ifstream if_pose(pose_file_path);

	Matrix3d Rwc1;
	Matrix3d Rwc2;
	Quaterniond q_wc1;
	Quaterniond q_wc2;

	Vector3d twc1;
	Vector3d twc2;

	std::string data_line_pose;
	string tmp;

//	Twc1
	std::getline(if_pose, data_line_pose);
	std::istringstream poseData1(data_line_pose);
	poseData1 >> tmp >> twc1.x() >> twc1.y() >> twc1.z()
			 >> q_wc1.x() >> q_wc1.y() >> q_wc1.z() >> q_wc1.w();
	Rwc1 = q_wc1.matrix();

//	Twc2
	std::getline(if_pose, data_line_pose);
	std::istringstream poseData2(data_line_pose);
	poseData2 >> tmp >> twc2.x() >> twc2.y() >> twc2.z()
			 >> q_wc2.x() >> q_wc2.y() >> q_wc2.z() >> q_wc2.w();
	Rwc2 = q_wc2.matrix();

	if_pose.close();

	cout << "twc1:  " << twc1.transpose() << endl;
	cout << "twc2:  " << twc2.transpose() << endl;
	cout << "q_wc1: " << q_wc1.coeffs().transpose() << endl;
	cout << "q_wc2: " << q_wc2.coeffs().transpose() << endl;

//	计算帧间相对位姿
	Matrix3d Rc1c2;
	Vector3d tc1c2;

	Matrix3d Rc2c1;
	Vector3d tc2c1;

	Rc1c2 = Rwc1.transpose() * Rwc2;
	tc1c2 = Rwc1.transpose() * twc2 - Rwc1.transpose() * twc1;
	Rc2c1 = Rc1c2.inverse();
	tc2c1 = -Rc2c1 * tc1c2;

	cout<<"t_c1c2: "<<tc1c2.transpose()<<endl;

//	读取深度图
	cv::Mat depth1 = imread(path + "/depth1.png", CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
	cv::Mat depth2 = imread(path + "/depth2.png", CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
//	读取彩色图
	cv::Mat rgb1 = imread(path+"/rgb1.png", CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat rgb2 = imread(path+"/rgb2.png", CV_LOAD_IMAGE_UNCHANGED);

//	用作显示的图像
	cv::Mat feature_1 = rgb1.clone();
	cv::Mat feature_2 = rgb2.clone();

//	转为灰度图
	cv::Mat gray1;
	cv::Mat gray2;
	cvtColor(rgb1,gray1,CV_RGB2GRAY);
	cvtColor(rgb2,gray2,CV_RGB2GRAY);

//	将当前图像通过位姿投影到参考帧中
	double depth_factor = 5000.0;
	cv::Mat depth_curr_prj2_ref;

//	将当前帧深度投影到参考帧，生成深度图
	depth_prj(depth_factor, K,
			  Rc1c2, tc1c2,
			  depth1, depth2, depth_curr_prj2_ref);

//	特征提取：

	vector<Point2i> kpts_1;
	vector<Point2i> kpts_2;

	extr_match_kpts_cv(
		rgb1,
		rgb2,
		kpts_1,
		kpts_2
	);

//	绘制对应的特征点
	Scalar color(0,0,255);
	drawKeyPoints_num(color,feature_1,kpts_1);
	drawKeyPoints_num(color,feature_2,kpts_2);

//	特征点反投影，查看投影精度,获取深度真值
	std::vector<Point2f> c2_prj2_c1_pts;

	for (int i = 0; i < kpts_2.size(); ++i) {
		int r = kpts_2[i].y;
		int c = kpts_2[i].x;

		while(depth2.at<char16_t>(r, c) == 0){
			r++;
		}

		double depth_c2 = (double)depth2.at<char16_t>(r, c) / depth_factor;

		Vector3d P_c2 = depth_c2 * K_inv * Vector3d(c, r, 1);
		Vector3d P_c1 = Rc1c2 * P_c2 + tc1c2;
		Vector3d p_c1_prj = K * (P_c1 / P_c1.z());

//			如果还在图像内，进行投影，生成新的深度图
		if (p_c1_prj.x() >= 0 && p_c1_prj.x() <= width
			&& p_c1_prj.y() >= 0 && p_c1_prj.y() <= height
			) {
			c2_prj2_c1_pts.push_back(Point2f(p_c1_prj.x(), p_c1_prj.y()));
		}
	}

	color = Scalar(0,255,0);
	drawKeyPoints_num(color, feature_1, c2_prj2_c1_pts);

//	梯度图像，梯度模长，x梯度，y梯度
	cv::Mat gradient_length1;
	cv::Mat gradient_x1;
	cv::Mat gradient_y1;

	cv::Mat gradient_length2;
	cv::Mat gradient_x2;
	cv::Mat gradient_y2;


//	计算梯度
	calc_gradient(gray1,gradient_length1,gradient_x1,gradient_y1);
	calc_gradient(gray2,gradient_length2,gradient_x2,gradient_y2);

//	对c1下的第0个特征点计算极线,判断点的有效性

	float depth_est = 10.0;
	std::vector<float> point_depth_gt1;
	std::vector<float> point_depth_gt2;
	float error_rate = 0;
	for (int j = 0; j < kpts_1.size(); ++j) {

		point_depth_gt1.push_back(depth1.at<char16_t>(kpts_1[j]) / depth_factor);
		point_depth_gt2.push_back(depth2.at<char16_t>(kpts_2[j]) / depth_factor);
		cout<<"depth ground truth: [c1:c2] = ["<< point_depth_gt1.back()<<":"<<point_depth_gt2.back()<<"]"<<endl;

		Point2f p_c1 = kpts_1[j];
		Vector2f this_gradient1(gradient_x1.at<float>(p_c1),gradient_y1.at<float>(p_c1));
		float this_gradient_length1 = gradient_length1.at<float>(p_c1);

		Vector2f epiline;
		epiline.x() = - fx * tc1c2.x() + tc1c2.z() * (p_c1.x - cx);
		epiline.y() = - fy * tc1c2.y() + tc1c2.z() * (p_c1.y - cy);
		float  epiline_length = epiline.norm();  // 若epiline_length太小 debug1
		epiline = epiline / epiline_length;

		float gradient_alone_epipolar = fabs(this_gradient1.dot(epiline)); //若极线的方向上梯度投影<2 ，debug2

		cout << "point "<< j << "=======================" << endl;
		cout << "this_gradient1: " << this_gradient1.transpose() << endl;
		cout << "epiline: " << epiline.transpose() << endl;
		cout << "epiline_length: " << epiline_length << endl;
		cout << "gradient_alone_epipolar: " << gradient_alone_epipolar << endl;
		cout << "cos angular: " << gradient_alone_epipolar/this_gradient_length1 << endl;  //若cos极线梯度夹角 < 0.3 ，debug3

//	计算当前帧上的搜索范围
		float search_min_idep = MIN_INV_DEPTH;
		float search_max_idep = MAX_INV_DEPTH;

		float best_idepth;
		float alpha;

		search_point(
			K,
			Rc2c1,
			tc2c1,
			(int)p_c1.x,
			(int)p_c1.y,
			search_max_idep,
			search_min_idep,
			epiline,
			gray1,
			gray2,
			feature_2,
			best_idepth,
			alpha
		);
		error_rate += abs((1.0f / best_idepth) - point_depth_gt1[j]) / point_depth_gt1[j];

		cout << "estmated depth = " << 1.0f / best_idepth << endl;
		cout << "gt       depth = " << point_depth_gt1[j] << endl;

	}

	cout << "error rate = " << error_rate / point_depth_gt1.size() << endl;

//	显示图像
	gradient_x1.convertTo(gradient_x1,CV_8UC1);
	imshow("feature_1",feature_1);
	imshow("feature_2",feature_2);



	cv::waitKey(0);
}


