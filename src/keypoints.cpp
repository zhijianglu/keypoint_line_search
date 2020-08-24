//
// Created by lab on 2020/8/24.
//

#include "keypoints.h"

void extr_match_kpts_cv(
	Mat& img_1,
	Mat& img_2,
	vector<Point2i>& kpts_1,
	vector<Point2i>& kpts_2
	){

	std::vector<KeyPoint> keypoints_1;
	std::vector<KeyPoint> keypoints_2;

	Mat descriptors_1,descriptors_2;
	extract_feature(img_1,keypoints_1,descriptors_1);
	extract_feature(img_2,keypoints_2,descriptors_2);

	selectPt_and_match(
		10.0,
		keypoints_1,kpts_1, descriptors_1,
		keypoints_2,kpts_2, descriptors_2
	);

}

void selectPt_and_match(
	double min_distance,
	 std::vector<KeyPoint> &keypoints_1,vector<Point2i>& kpts_1, Mat &descriptors_1,
	 std::vector<KeyPoint> &keypoints_2,vector<Point2i>& kpts_2, Mat &descriptors_2)
{

	vector<DMatch> matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

	matcher->match(descriptors_1, descriptors_2, matches);

	//-- 第四步:匹配点对筛选
	double min_dist = 10000, max_dist = 0;

	//找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
	for (int i = 0; i < descriptors_1.rows; i++) {
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	min_dist = min_element(matches.begin(), matches.end(), [](const DMatch &m1, const DMatch &m2)
	{ return m1.distance < m2.distance; })->distance;
	max_dist = max_element(matches.begin(), matches.end(), [](const DMatch &m1, const DMatch &m2)
	{ return m1.distance < m2.distance; })->distance;

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.

	int cnt = 0;

	std::vector<DMatch> good_matches;
	std::vector<DMatch> good_matches_my;
	for (int i = 0; i < descriptors_1.rows; i++) {
		if (matches[i].distance <= max(2 * min_dist, min_distance)) {
			good_matches.push_back(matches[i]);
			kpts_1.push_back(keypoints_1[i].pt);
			kpts_2.push_back(keypoints_2[matches[i].trainIdx].pt);
			DMatch match = matches[i];
			match.trainIdx = cnt;
			match.queryIdx = cnt;
			good_matches_my.push_back(match);
			cnt++;
		}
	}

//	滤除相隔比较近的点
	for (int j = 0; j < kpts_1.size(); ++j) {
		for (int k = j + 1; k < kpts_1.size(); ++k) {
			Point2f diff_array = kpts_1[j] - kpts_1[k];
//			cout << "  " << diff_array.dot(diff_array) << endl;
			if (diff_array.dot(diff_array) < 800.0) {
				kpts_1.erase(kpts_1.begin() + k);
				kpts_2.erase(kpts_2.begin() + k);
			}
		}
	}

	//-- 第五步:绘制匹配结果
//	Mat img_goodmatch;
//	drawMatches ( img_1, goodpoint_1, img_2, goodpoint_2, good_matches_my, img_goodmatch );
//	imshow ( "opted_match", img_goodmatch );
//	waitKey(0);

}


void extract_feature(Mat &input_mat, std::vector<KeyPoint> &keypoints, Mat &descriptors)
{

	//-- 初始化
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();

	//-- 第一步:检测 Oriented FAST 角点位置
	detector->detect(input_mat, keypoints);

	//-- 第二步:根据角点位置计算 BRIEF 描述子
	descriptor->compute(input_mat, keypoints, descriptors);

}