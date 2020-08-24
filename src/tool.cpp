//
// Created by lab on 2020/8/23.
//

#include "tool.h"



void depth_prj(double depth_factor, Matrix3d &K,
			   Matrix3d &Rrc, Vector3d &trc,
			   Mat &depth_ref, Mat &depth_curr, Mat &depth_curr_prj2_ref)
{
	int height = depth_ref.rows;
	int width = depth_ref.cols;
	depth_curr_prj2_ref = cv::Mat(height, width, CV_16UC1, Scalar(0));
	Matrix3d K_inv = K.inverse();

	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			if (depth_curr.at<char16_t>(r, c) == 0) continue;
			double depth_c2 = (double)depth_curr.at<char16_t>(r, c) / depth_factor;
			Vector3d P_c2 = depth_c2 * K_inv * Vector3d(c, r, 1);
			Vector3d P_c1 = Rrc * P_c2 + trc;
			Vector3d p_c1_prj = K * (P_c1 / P_c1.z());

//			如果还在图像内，进行投影，生成新的深度图
			if (p_c1_prj.x() >= 0 && p_c1_prj.x() <= width
				&& p_c1_prj.y() >= 0 && p_c1_prj.y() <= height
				) {
				depth_curr_prj2_ref.at<char16_t>(p_c1_prj.y(), p_c1_prj.x()) = P_c1.z() * depth_factor;
			}
		}
	}
}



void drawKeyPoints_num(
	Scalar color,
	Mat & img,std::vector<Point2i>& keypoints
	){
	for (int i = 0; i < keypoints.size(); ++i) {
//		Scalar color(25 * i % 255, 255-(25 * i % 255),125-(25 * i % 125));

		cv::drawMarker(img, keypoints[i], color, 3, 6, 1);
		cv::putText(img, to_string(i), keypoints[i] + Point2i(-4,-4), 1, 1.5, color, 2);
	}
}

void drawKeyPoints_num(
	Scalar color,
	Mat & img,std::vector<Point2f>& keypoints
){
	for (int i = 0; i < keypoints.size(); ++i) {
//		cv::drawMarker(img, keypoints[i], color, (i+1) % 7, 4, 3);
		cv::circle(img,keypoints[i],1,color,2,5);
		cv::putText(img, to_string(i), keypoints[i] + Point2f(8,+8), 1, 1.5, color, 2);
	}
}

void calc_gradient(Mat &img, Mat &gradient_length,Mat &gradient_x,Mat &gradient_y)
{
	gradient_length = Mat(img.size(), CV_32FC1, Scalar(0));
	gradient_x = Mat(img.size(), CV_32FC1, Scalar(0));
	gradient_y = Mat(img.size(), CV_32FC1, Scalar(0));
	float dx = 0;
	float dy = 0;
	int width = img.cols;
	int height = img.rows;
	for (int c = 0; c < width; ++c) {
		for (int r = 0; r < height; ++r) {
			int up = r == 0 ? r : (r - 1);
			int down = r == (height-1) ? r : r+1;

			int left = c == 0 ? c : (c - 1);
			int right = c == width-1 ? c : (c + 1);

			dx = ((float)img.at<uchar>(r,right) - (float)img.at<uchar>(r,left))/2.0f;
			dy = ((float)img.at<uchar>(down,c) - (float)img.at<uchar>(up,c))/2.0f;

			gradient_length.at<float>(r, c) = sqrtf(dx * dx + dy * dy);
			gradient_x.at<float>(r, c) = dx;
			gradient_y.at<float>(r, c) = dy;
		}
	}
}

void search_point(
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
	){

	int width = keyframe_image_tex.cols;
	int height = keyframe_image_tex.rows;

//	获取到参考帧上基线方向的周边几个点
	float this_patch[5];
#pragma unroll
	for(int i = -2; i <= 2; i++)  //关键帧上的待投影点 this_patch = [ ○ ○ ※ ○ ○ ]  ※为当前投影点
		this_patch[i+2] = keyframe_image_tex.at<uchar>(y + i * epipolar_line.y() + 0.5,x + i * epipolar_line.x() + 0.5);

//	计算在当前图像上基于基线方向上的搜索范围
//	step1. 先将参考帧的像素点投影到当前坐标系下，根据最近和最远深度投影，确定搜索范围
	Vector3d Ki_pr = K.inverse() * Vector3d(x,y,1);
	Vector3d Rcr_Ki_pr = R_cr * Ki_pr;
	Vector3d K_Rcr_Ki_pr = K * Rcr_Ki_pr;
	Vector3d K_tcr = K * t_cr;

	Vector3d near_point  = K_Rcr_Ki_pr + K_tcr * max_idep;
	Vector3d far_point   = K_Rcr_Ki_pr + K_tcr * min_idep;

//	step2. 将当前帧坐标系下的两个点分别投影到当前帧图像上，得到搜索像素范围
	Vector2d near_uv(near_point.x() / near_point.z(), near_point.y() / near_point.z());
	Vector2d far_uv(far_point.x() / far_point.z(), far_point.y() / far_point.z());
	float search_length =(near_uv-far_uv).norm();
	Vector2d search_dir = (near_uv-far_uv) / search_length;  //搜索方向向量


//	step3. 有了搜索范围,将搜索范围内的点强度值进行比对，争取得到最小光度误差的匹配
	float that_patch[5];
#pragma unroll  //将代码展开 使得循环更加高效
	for(int i = -2; i <= 1; i++)  // [ ○ ○ ※ ○ ] 先获取当前帧搜索线上起始点的左边两个右边一个点
		that_patch[i+2] = currframe_image_tex.at<uchar>(far_uv.y() + i * search_dir.y() + 0.5,far_uv.x() + i * search_dir.x() + 0.5);
	int step = 0;
	int MAX_SEARCH = 100;
	float best_step = -1;
	float best_score = 1e9;
	float pre_best_score = 1e9;
	float post_best_score = 1e9;
	bool last_is_best = false;

	// best pre and post errors.
	float best_match_DiffErrPre=NAN, best_match_DiffErrPost=NAN;
	// alternating intermediate vars
	float e1A=NAN, e1B=NAN, e2A=NAN, e2B=NAN, e3A=NAN, e3B=NAN, e4A=NAN, e4B=NAN, e5A=NAN, e5B=NAN;

	float second_best_score = 1e9;  //第二小的匹配光度误差
	float second_best_step = -1;    //第二小的误差对应的步长，或者说像素位置

	float pre_score = 1e9;  //for loop use 当前误差的相邻点（上一个点）的误差
	float this_score = 1e9; //pitch内的五个点的光度误差平方和，越小越好

	//TODO 开始搜索 ！！匹配点----------------------------------------------
	for (; step <= search_length && step <= MAX_SEARCH; step += 1) {
		pre_score = this_score;
		this_score = 0.0f;

		Vector2d search_point = far_uv + step * search_dir;

//		越界放弃
		if (search_point.x() <= 1 || search_point.x() >= width - 1 || search_point.y() <= 1
			|| search_point.y() >= height - 1)
			break;

//		更新patch，循环移动
		that_patch[4] = currframe_image_tex
			.at<uchar>(search_point.y() + 2.0 * search_dir.y() + 0.5, search_point.x() + 2.0 * search_dir.x() + 0.5);

		if (step % 2 == 0) //A 0 2 4 6 ... 偶数步长
		{
			e1A = that_patch[4] - this_patch[4];
			this_score += e1A * e1A;
			e2A = that_patch[3] - this_patch[3];
			this_score += e2A * e2A;
			e3A = that_patch[2] - this_patch[2];
			this_score += e3A * e3A;  // 此为当前搜索点
			e4A = that_patch[1] - this_patch[1];
			this_score += e4A * e4A;
			e5A = that_patch[0] - this_patch[0];
			this_score += e5A * e5A;
		}
		else  //B 1 3 5 7 ... 奇数步长
		{
			e1B = that_patch[4] - this_patch[4];
			this_score += e1B * e1B;
			e2B = that_patch[3] - this_patch[3];
			this_score += e2B * e2B;
			e3B = that_patch[2] - this_patch[2];
			this_score += e3B * e3B;
			e4B = that_patch[1] - this_patch[1];
			this_score += e4B * e4B;
			e5B = that_patch[0] - this_patch[0];
			this_score += e5B * e5B;
		}

		if (last_is_best) {
			post_best_score = this_score;
			best_match_DiffErrPost = e1A * e1B + e2A * e2B + e3A * e3B + e4A * e4B + e5A * e5B;
			last_is_best = false;
		}
		if (this_score < best_score) {
			//chane the best to second
			second_best_score = best_score;
			second_best_step = best_step;

			best_score = this_score;
			best_step = step;
			pre_best_score = pre_score;
			best_match_DiffErrPre = e1A * e1B + e2A * e2B + e3A * e3B + e4A * e4B + e5A * e5B;//patch内的搜索点差分误差
			best_match_DiffErrPost = -1;
			post_best_score = 1e9;
			last_is_best = true;
		}
		else if (this_score < second_best_score) {
			second_best_score = this_score;
			second_best_step = step;
		}

		that_patch[0] = that_patch[1];
		that_patch[1] = that_patch[2];
		that_patch[2] = that_patch[3];
		that_patch[3] = that_patch[4];
	}

//	筛选匹配
	float best_match_sub = best_step;
	bool didSubpixel = false;
	{
		// ================== compute exact match =========================
		// compute gradients (they are actually only half the real gradient)
		float gradPre_pre = -(pre_best_score - best_match_DiffErrPre);
		float gradPre_this = +(best_score - best_match_DiffErrPre);

		float gradPost_this = -(best_score - best_match_DiffErrPost);
		float gradPost_post = +(post_best_score - best_match_DiffErrPost);

		// final decisions here.
		bool interpPost = false;
		bool interpPre = false;

		// if pre has zero-crossing
		if((gradPre_pre < 0) ^ (gradPre_this < 0))  // 01或者10 即为true
		{
			// if post has zero-crossing
			if( (gradPost_post < 0) ^ (gradPost_this < 0) )
			{
			}
			else
				interpPre = true;
		}
			// if post has zero-crossing  有过零点
		else if((gradPost_post < 0) ^ (gradPost_this < 0))
		{
			interpPost = true;
		}


		// DO interpolation!插值
		// minimum occurs at zero-crossing of gradient, which is a straight line => easy to compute.
		// the error at that point is also computed by just integrating.
		if(interpPre)
		{
			float d = gradPre_this / (gradPre_this - gradPre_pre);
			best_match_sub -= d;
			best_score = best_score - 2*d*gradPre_this - (gradPre_pre - gradPre_this)*d*d;
			didSubpixel = true;
		}
		else if(interpPost)
		{
			float d = gradPost_this / (gradPost_this - gradPost_post);
			best_match_sub -= d;
			best_score = best_score + 2*d*gradPost_this + (gradPost_post - gradPost_this)*d*d;
			didSubpixel = true;
		}
	}

	float gradAlongLine = 0;
#pragma unroll
	for(int i = 1; i < 5; i++)
		gradAlongLine += (this_patch[i] - this_patch[i-1])*(this_patch[i] - this_patch[i-1]);
	float MAX_ERROR_STEREO = 1300.0f;

	if(best_score > MAX_ERROR_STEREO + sqrtf(gradAlongLine) * 20)  //搜索点patch梯度太大，依旧放弃，什么原理？？
		cout<<" ******** "<< endl;

//	计算深度
	if(search_dir.x()*search_dir.x() > search_dir.y()*search_dir.y())
	{
		float best_u = far_uv.x() + best_match_sub * search_dir.x();
		best_idepth = (best_u * K_Rcr_Ki_pr.z() - K_Rcr_Ki_pr.x()) / (K_tcr.x() - best_u*K_tcr.z());
		alpha = (K_Rcr_Ki_pr.z()*K_tcr.x() - K_tcr.z()*K_Rcr_Ki_pr.x()) / ((K_tcr.x() - best_u*K_tcr.z()) * (K_tcr.x() - best_u*K_tcr.z()));
	}
	else
	{
		float best_v = far_uv.y() + best_match_sub * search_dir.y();
		best_idepth = (best_v * K_Rcr_Ki_pr.z() - K_Rcr_Ki_pr.y())/(K_tcr.y() - best_v*K_tcr.z());
		alpha = (K_Rcr_Ki_pr.z()*K_tcr.y() - K_tcr.z()*K_Rcr_Ki_pr.y()) / ((K_tcr.y() - best_v*K_tcr.z()) * (K_tcr.y() - best_v*K_tcr.z()));
	}
	best_idepth = best_idepth < min_idep ? min_idep : best_idepth;
	best_idepth = best_idepth > max_idep ? max_idep : best_idepth;


//	绘制结果
	float best_u = far_uv.x() + best_score * search_dir.x();
	float best_v = far_uv.y() + best_score * search_dir.y();
	cv::circle(result_show,Point2i(best_u,best_v),1,Scalar(0,255,255),2,5);

	best_u = far_uv.x() + best_match_sub * search_dir.x();
	best_v = far_uv.y() + best_match_sub * search_dir.y();

	cv::line(result_show,Point2i(near_uv.x(),near_uv.y()),Point2i(far_uv.x(),far_uv.y()),Scalar(255),1);
//	cv::circle(result_show,Point2i(best_u,best_v),1,Scalar(0,255,0),2,5);
	cv::drawMarker(result_show, Point2i(best_u,best_v), Scalar(0,255,0), 3, 6, 1);

}

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

){
	int width = keyframe_image_tex.cols;
	int height = keyframe_image_tex.rows;

//	获取到参考帧上基线方向的周边几个点
	float this_patch[5];
	float this_up_patch[3];
	float this_down_patch[3];
#pragma unroll
	for(int i = -2; i <= 2; i++)  //关键帧上的待投影点 this_patch = [ ○ ○ ※ ○ ○]  ※为当前投影点
		this_patch[i+2] = keyframe_image_tex.at<uchar>(y + i * epipolar_line.y() + 0.5,x + i * epipolar_line.x() + 0.5);

#pragma unroll
	for (int i = -1; i <= 1; i++)  //关键帧上的待投影点 this_patch = [ ○ ※ ○]  ※为当前投影点
		this_up_patch[i + 1] =
			keyframe_image_tex.at<uchar>(y + 1 + i * epipolar_line.y() + 0.5, x + i * epipolar_line.x() + 0.5);

#pragma unroll
	for (int i = -1; i <= 1; i++)  //关键帧上的待投影点 this_patch = [ ○ ※ ○ ]  ※为当前投影点
		this_down_patch[i + 1] =
			keyframe_image_tex.at<uchar>(y - 1 + i * epipolar_line.y() + 0.5, x + i * epipolar_line.x() + 0.5);



//	计算在当前图像上基于基线方向上的搜索范围
//	step1. 先将参考帧的像素点投影到当前坐标系下，根据最近和最远深度投影，确定搜索范围
	Vector3d Ki_pr = K.inverse() * Vector3d(x,y,1);
	Vector3d Rcr_Ki_pr = R_cr * Ki_pr;
	Vector3d K_Rcr_Ki_pr = K * Rcr_Ki_pr;
	Vector3d K_tcr = K * t_cr;

	Vector3d near_point  = K_Rcr_Ki_pr + K_tcr * max_idep;
	Vector3d far_point   = K_Rcr_Ki_pr + K_tcr * min_idep;

//	step2. 将当前帧坐标系下的两个点分别投影到当前帧图像上，得到搜索像素范围
	Vector2d near_uv(near_point.x() / near_point.z(), near_point.y() / near_point.z());
	Vector2d far_uv(far_point.x() / far_point.z(), far_point.y() / far_point.z());
	float search_length =(near_uv-far_uv).norm();
	Vector2d search_dir = (near_uv-far_uv) / search_length;  //搜索方向向量


//	step3. 有了搜索范围,将搜索范围内的点强度值进行比对，争取得到最小光度误差的匹配
	float that_patch[5];
	float that_up_patch[3];
	float that_down_patch[3];
#pragma unroll  //将代码展开 使得循环更加高效
	for (int i = -2; i <= 1; i++)  // [ ○ ○ ※ ○ ] 先获取当前帧搜索线上起始点的左边两个右边一个点
		that_patch[i + 2] =
			currframe_image_tex.at<uchar>(far_uv.y() + i * search_dir.y() + 0.5, far_uv.x() + i * search_dir.x() + 0.5);

#pragma unroll  //将代码展开 使得循环更加高效
	for (int i = -1; i <= 0; i++)  // [ ○ ※ ] 先获取当前帧搜索线上起始点的左边两个右边一个点
		that_up_patch[i + 1] =
			currframe_image_tex.at<uchar>(far_uv.y()+1 + i * search_dir.y() + 0.5, far_uv.x() + i * search_dir.x() + 0.5);

#pragma unroll  //将代码展开 使得循环更加高效
	for (int i = -1; i <= 0; i++)  // [ ○ ※ ] 先获取当前帧搜索线上起始点的左边两个右边一个点
		that_down_patch[i + 1] =
			currframe_image_tex.at<uchar>(far_uv.y()-1 + i * search_dir.y() + 0.5, far_uv.x() + i * search_dir.x() + 0.5);


	int step = 0;
	int MAX_SEARCH = 100;
	float best_step = -1;
	float best_score = 1e9;
	float pre_best_score = 1e9;
	float post_best_score = 1e9;
	bool last_is_best = false;

	// best pre and post errors.
	float best_match_DiffErrPre=NAN, best_match_DiffErrPost=NAN;
	// alternating intermediate vars
	float e1A=NAN, e1B=NAN, e2A=NAN, e2B=NAN, e3A=NAN, e3B=NAN, e4A=NAN, e4B=NAN, e5A=NAN, e5B=NAN;

	float second_best_score = 1e9;  //第二小的匹配光度误差
	float second_best_step = -1;    //第二小的误差对应的步长，或者说像素位置

	float pre_score = 1e9;  //for loop use 当前误差的相邻点（上一个点）的误差
	float this_score = 1e9; //pitch内的五个点的光度误差平方和，越小越好

	//TODO 开始搜索 ！！匹配点----------------------------------------------
	for (; step <= search_length && step <= MAX_SEARCH; step += 1) {
		pre_score = this_score;
		this_score = 0.0f;

		Vector2d search_point = far_uv + step * search_dir;

//		越界放弃
		if (search_point.x() <= 1 || search_point.x() >= width - 1 || search_point.y() <= 1
			|| search_point.y() >= height - 1)
			break;

//		更新patch，循环移动
		that_patch[4] = currframe_image_tex
			.at<uchar>(search_point.y() + 2.0 * search_dir.y() + 0.5, search_point.x() + 2.0 * search_dir.x() + 0.5);

		that_up_patch[2] = currframe_image_tex
			.at<uchar>(search_point.y() + 2.0 * search_dir.y() + 0.5, search_point.x() + 1.0 * search_dir.x() + 0.5);

		that_down_patch[2] = currframe_image_tex
			.at<uchar>(search_point.y() + 0.0 * search_dir.y() + 0.5, search_point.x() + 1.0 * search_dir.x() + 0.5);


		if (step % 2 == 0) //A 0 2 4 6 ... 偶数步长
		{
			e1A = that_patch[4] - this_patch[4];
			this_score += e1A * e1A;

			e2A =
				(that_patch[3] - this_patch[3]) +
					(that_up_patch[2] - this_up_patch[2]) +
					(that_down_patch[2] - this_down_patch[2]);
			this_score += e2A * e2A;

			e3A = that_patch[2] - this_patch[2] +
				(that_up_patch[1] - this_up_patch[1]) +
				(that_down_patch[1] - this_down_patch[1]);
			this_score += e3A * e3A;  // 此为当前搜索点

			e4A = that_patch[1] - this_patch[1]+
				(that_up_patch[0] - this_up_patch[0]) +
				(that_down_patch[0] - this_down_patch[0]);
			this_score += e4A * e4A;

			e5A = that_patch[0] - this_patch[0];
			this_score += e5A * e5A;
		}
		else  //B 1 3 5 7 ... 奇数步长
		{
			e1B = that_patch[4] - this_patch[4];
			this_score += e1B * e1B;

			e2B = that_patch[3] - this_patch[3] +
				(that_up_patch[2] - this_up_patch[2]) +
				(that_down_patch[2] - this_down_patch[2]);
			this_score += e2B * e2B;

			e3B = that_patch[2] - this_patch[2] +
				(that_up_patch[1] - this_up_patch[1]) +
				(that_down_patch[1] - this_down_patch[1]);
			this_score += e3B * e3B;

			e4B = that_patch[1] - this_patch[1]+
				(that_up_patch[0] - this_up_patch[0]) +
				(that_down_patch[0] - this_down_patch[0]);
			this_score += e4B * e4B;

			e5B = that_patch[0] - this_patch[0];
			this_score += e5B * e5B;
		}

		if (last_is_best) {
			post_best_score = this_score;
			best_match_DiffErrPost = e1A * e1B + e2A * e2B + e3A * e3B + e4A * e4B + e5A * e5B;
			last_is_best = false;
		}
		if (this_score < best_score) {
			//chane the best to second
			second_best_score = best_score;
			second_best_step = best_step;

			best_score = this_score;
			best_step = step;
			pre_best_score = pre_score;
			best_match_DiffErrPre = e1A * e1B + e2A * e2B + e3A * e3B + e4A * e4B + e5A * e5B;//patch内的搜索点差分误差
			best_match_DiffErrPost = -1;
			post_best_score = 1e9;
			last_is_best = true;
		}
		else if (this_score < second_best_score) {
			second_best_score = this_score;
			second_best_step = step;
		}

		that_patch[0] = that_patch[1];
		that_patch[1] = that_patch[2];
		that_patch[2] = that_patch[3];
		that_patch[3] = that_patch[4];

		that_up_patch[1] = that_up_patch[2];
		that_up_patch[0] = that_up_patch[1];

		that_down_patch[1] = that_down_patch[2];
		that_down_patch[0] = that_down_patch[1];

	}

//	筛选匹配
	float best_match_sub = best_step;
	bool didSubpixel = false;
	{
		// ================== compute exact match =========================
		// compute gradients (they are actually only half the real gradient)
		float gradPre_pre = -(pre_best_score - best_match_DiffErrPre);
		float gradPre_this = +(best_score - best_match_DiffErrPre);

		float gradPost_this = -(best_score - best_match_DiffErrPost);
		float gradPost_post = +(post_best_score - best_match_DiffErrPost);

		// final decisions here.
		bool interpPost = false;
		bool interpPre = false;

		// if pre has zero-crossing
		if((gradPre_pre < 0) ^ (gradPre_this < 0))  // 01或者10 即为true
		{
			// if post has zero-crossing
			if( (gradPost_post < 0) ^ (gradPost_this < 0) )
			{
			}
			else
				interpPre = true;
		}
			// if post has zero-crossing  有过零点
		else if((gradPost_post < 0) ^ (gradPost_this < 0))
		{
			interpPost = true;
		}


		// DO interpolation!插值
		// minimum occurs at zero-crossing of gradient, which is a straight line => easy to compute.
		// the error at that point is also computed by just integrating.
		if(interpPre)
		{
			float d = gradPre_this / (gradPre_this - gradPre_pre);
			best_match_sub -= d;
			best_score = best_score - 2*d*gradPre_this - (gradPre_pre - gradPre_this)*d*d;
			didSubpixel = true;
		}
		else if(interpPost)
		{
			float d = gradPost_this / (gradPost_this - gradPost_post);
			best_match_sub -= d;
			best_score = best_score + 2*d*gradPost_this + (gradPost_post - gradPost_this)*d*d;
			didSubpixel = true;
		}
	}

	float gradAlongLine = 0;
#pragma unroll
	for(int i = 1; i < 5; i++)
		gradAlongLine += (this_patch[i] - this_patch[i-1])*(this_patch[i] - this_patch[i-1]);
	float MAX_ERROR_STEREO = 2166.0f;

	if(best_score > MAX_ERROR_STEREO + sqrtf(gradAlongLine) * 20)  //搜索点patch梯度太大，依旧放弃，什么原理？？
		cout<<" ******** "<< endl;

//	计算深度
	if(search_dir.x()*search_dir.x() > search_dir.y()*search_dir.y())
	{
		float best_u = far_uv.x() + best_match_sub * search_dir.x();
		best_idepth = (best_u * K_Rcr_Ki_pr.z() - K_Rcr_Ki_pr.x()) / (K_tcr.x() - best_u*K_tcr.z());
		alpha = (K_Rcr_Ki_pr.z()*K_tcr.x() - K_tcr.z()*K_Rcr_Ki_pr.x()) / ((K_tcr.x() - best_u*K_tcr.z()) * (K_tcr.x() - best_u*K_tcr.z()));
	}
	else
	{
		float best_v = far_uv.y() + best_match_sub * search_dir.y();
		best_idepth = (best_v * K_Rcr_Ki_pr.z() - K_Rcr_Ki_pr.y())/(K_tcr.y() - best_v*K_tcr.z());
		alpha = (K_Rcr_Ki_pr.z()*K_tcr.y() - K_tcr.z()*K_Rcr_Ki_pr.y()) / ((K_tcr.y() - best_v*K_tcr.z()) * (K_tcr.y() - best_v*K_tcr.z()));
	}
	best_idepth = best_idepth < min_idep ? min_idep : best_idepth;
	best_idepth = best_idepth > max_idep ? max_idep : best_idepth;
	

//	绘制匹配结果
	float best_u = far_uv.x() + best_score * search_dir.x();
	float best_v = far_uv.y() + best_score * search_dir.y();
	cv::circle(result_show,Point2i(best_u,best_v),1,Scalar(255,0,255),2,5);

	best_u = far_uv.x() + best_match_sub * search_dir.x();
	best_v = far_uv.y() + best_match_sub * search_dir.y();

//	cout<<"near_uv :"<<near_uv<<endl;
//	cout<<"far_uv  :"<<far_uv<<endl;

	cv::line(result_show, Point2i(near_uv.x(), near_uv.y()), Point2i(far_uv.x(), far_uv.y()), Scalar(255), 1);
//	cv::circle(result_show,Point2i(best_u,best_v),1,Scalar(0,255,0),2,5);
	cv::drawMarker(result_show, Point2i(best_u, best_v), Scalar(0, 255, 0), 3, 6, 1);

}