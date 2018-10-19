#include <vector>
#include <stdlib.h>
#include <stdio.h>

#include <string>
#include <opencv2/opencv.hpp>

#include "optimization.h"
#include "FundamentalMat.h"

using std::vector;
using cv::Mat_;
using cv::Mat;
using cv::Point2f;
using cv::Point3f;

const int NPROJ = 4; // replace with global setting

std::vector<std::vector<cv::Point2f>> m_cam_pts; // replace with displaycalibration member Make sure it is double
std::vector<std::vector<cv::Point2f>> m_proj_pts; // replace with displaycalibration member

void function_mincost(const Mat_<float>& epipole, Mat_<float>& A, std::vector<double>& err);

bool loadBlobData(const std::string& file_name, const int proj_idx)
{
	using namespace cv;
	Mat cam_blob, proj_blob;
	FileStorage fs(file_name, FileStorage::READ);

	if (!fs.isOpened())
	{
		throw std::runtime_error("loadBlobData() Failed to open " + file_name);
	}

	fs["cam_pts"] >> cam_blob;
	fs["proj_pts"] >> proj_blob;

	cam_blob.reshape(2).copyTo(m_cam_pts[proj_idx]);
	proj_blob.reshape(2).copyTo(m_proj_pts[proj_idx]);

	fs.release();
	return !m_cam_pts[proj_idx].empty();
}

int main(int argc, char **argv)
{
	// step 1: read xml files to m_cam_pts and m_proj_pts
	m_cam_pts.clear();
	m_cam_pts.resize(NPROJ);
	m_proj_pts.clear();
	m_proj_pts.resize(NPROJ);

	for (int i = 0; i<NPROJ; i++)
	{
		std::string file_name = "Proj" + std::to_string(i + 1) + "PairBlobData.xml";
		if (!loadBlobData(file_name, i))
		{
			std::cout << "zero vector found" << std::endl;
			return -1;
		}
	}
	// step 2: undistort camera distortion
	double p0[] = { 996.51215,1002.67977,512.06846,771.93374,1.08832,-0.06160,0.14964,-0.05711,0.27173,0.96068,996.51215,1002.67977,512.06846,771.93374,-1.05262,-0.29111,2.85308,0.63458,-0.23280,0.53617,996.51215,1002.67977,512.06846,771.93374,0.03536,1.66405,-2.55161,-0.02775,0.27802,1.02214,996.51215,1002.67977,512.06846,771.93374,0.17217,-0.71828,-0.09515,0.75570,-0.29859,0.66934,-0.04642,-0.05585,1.50006,1.14651,1022.77966,1028.77374,668.39546,507.42256,-0.37291,0.21417,-0.07881,0.00041,-0.00119};
	
	Mat_<float> cam_param = cv::Mat(9, 1, CV_32FC1);
	for (int k = 0; k < 9; k++)
		cam_param.at<float>(k) = static_cast<float> (p0[k + NPROJ * 10 + 4]);
	Mat_<float> cam_KK = (Mat_<float>(3, 3) << cam_param(0), 0.f, cam_param(2),
		0.f, cam_param(1), cam_param(3),
		0.f, 0.f, 1.f);
	Mat_<float> cam_dist = (Mat_<float>(5, 1) << cam_param(4), cam_param(5), cam_param(7), cam_param(8), cam_param(6));
	vector<vector<Point2f>> cam_pts_nml(4);
	std::cout << cam_KK << std::endl;
	for (int i = 0; i < NPROJ; i++)
	{
		cv::undistortPoints(m_cam_pts[i], cam_pts_nml[i], cam_KK, cam_dist);
		vector<Point3f> pts_h;
		convertPointsToHomogeneous(cam_pts_nml[i], pts_h);
		int numPts = pts_h.size();
		for (int j = 0; j < numPts; j++)
		{
			Mat_<float> temp = (Mat_<float>(3, 1) << pts_h[j].x, pts_h[j].y, pts_h[j].z);		
			Mat_<float> result = cam_KK * temp;
			cam_pts_nml[i][j] = Point2f(result(0), result(1));
		}
	}
	// step 3: initial guess of e
	stereo_recon::FundamentalMat fmat;

	int numPts = cam_pts_nml[0].size();

	fmat.setThresholdRatio(.99f);
	Mat F = fmat.findFundamentalMat(cam_pts_nml[0], m_proj_pts[0], stereo_recon::Sampson);
	std::cout << F << std::endl;

	Mat S, U, Vt;
	cv::SVD::compute(F, S, U, Vt);

	Mat_<float> e0 = Vt.row(2);
	std::cout << e0 << std::endl;
	
	vector<Mat> cam_channels(2);
	vector<Mat> proj_channels(2);
	cv::split(cam_pts_nml[0], cam_channels);
	cv::split(m_proj_pts[0], proj_channels);

	Mat x1x2, x1y2, x2y1, y1y2; // per-element multiply
	x1x2 = cam_channels[0].mul(proj_channels[0]);
	y1y2 = cam_channels[1].mul(proj_channels[1]);
	x1y2 = cam_channels[0].mul(proj_channels[1]);
	x2y1 = cam_channels[1].mul(proj_channels[0]);
	
	Mat_<float> A;
	hconcat(x1x2.t(), x2y1.t(), A);
	hconcat(A, proj_channels[0].t(), A);
	hconcat(A, x1y2.t(), A);
	hconcat(A, y1y2.t(), A);
	hconcat(A, proj_channels[1].t(), A);
	hconcat(A, cam_channels[0].t(), A);
	hconcat(A, cam_channels[1].t(), A);
	hconcat(A, Mat::ones(numPts, 1, CV_32FC1), A);
	
	std::cout << cv::sum(cv::abs(A * F.reshape(1, 1).t())) << std::endl;
	
	// step 4: to-do optimize
	
	// test cost function

	// create optimizer without Jacobian
	
	// output error and e

	// get F back from e
	return 0;
}

void function_mincost(const Mat_<float>& epipole, Mat_<float>& A, std::vector<double>& err)
{

}
