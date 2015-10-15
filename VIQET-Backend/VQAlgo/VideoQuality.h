/*
Copyright © 2015 Intel Corporation
This program and the accompanying materials are made available under the terms of the Eclipse Public License v1.0, 
 which accompanies this distribution, and is available at http://www.eclipse.org/legal/epl-v10.html . https://github.com/viqet
* Contributors:
*    Intel Corporation - initial API and implementation and/or initial documentation
*/
#define NOMINMAX
#include<windows.h>

#include "svmUtils.h"

#include <iostream>
#include <cv.h>
#include <cmath>
#include <cxcore.h>
#include <highgui.h>
#include <contrib.hpp>
#include <fstream>
#include "FrameData.h"
#include "Constants.h"
#include <time.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <condition_variable>
#include "gamma.h"
#include <math.h>
#include <array>
#include <numeric> 

using namespace cv;
using namespace std;

class VideoQuality{
public:
	VideoQuality();
	~VideoQuality();
	static VideoQuality* pGetInstance();
	static string inputFile;
	FrameData* AnalyzePhoto(std::string& sourceFileName, std::string& category, std::string& mosModel, std::string& directory, bool onCloud, void* imgVoidPt = 0);
	vector<double> ValueSummary(vector<double> values);
	FrameData* getFrameData();
	mutex xmutex;  
	
	
	double Calculate_Saturation(void *imgVoidPt);
	static const void Calculate_Contrast(void *gimgVoidPt,double &Contrast_val);
	FrameData* Analyze_Frame(void *imgVoidPt);
	

	void Contrast_HeatMap(std::string& video_file);
	void Intensity_HeatMap(std::string& fileName, std::string& directory);
	void Sharpness_HeatMap(IplImage *img);
	bool DrawHistogram(IplImage *img);

private:

	static double resolution;
	int frame_cnt;
	CvCapture *capture;
	static const int size_N=7;
	static const int size_N96=96;
	static const int size_N48=48;
	static const int BLOCK_CONST=50;
	static const float thresh_hold;
	static const int MAX_KERNEL_LENGTH=3;
	static const int sharpness_algo_count=10;
	static const int sharpness_algo_count2=2;
	static const int MAX_KERNEL_LENGTH2=7;
	static const int size_N5=5;
	static const double ESP;
	static VideoQuality* _instance;
	bool first_write_to_file;
	vector<FrameData> frame_data;
	int blockCnt;

	float alpha,betaL,betaR,MeanParam;
	map<double,double> rgamMap;

	bool DayLight;

	unordered_map<String, double> numbers;
	map<int,unordered_map<String, double>>mapNumbers;
	
	
	 vector<float> CoherencyList;
	static double NF_1,NF_2,NF_3,NF_4,NF_5,NF_6,NF_7,NF_8,NF_9; 
	 Mat MatWithReduceValueForFinal9Scores;
	
	//thread
	static const void Calculate_Contrast(IplImage *gimg,double &Contrast_val);
	static const void Calculate_Saturation(IplImage *gimg,double &Saturation_val);
	static const void Calculate_Exposure(IplImage *gimg,double &Exposure_val);
	static const void Calculate_DynamicRange(IplImage *img,double &DynamicRange_val);
	static const void Calculate_Sharpness3(IplImage *img, double &Sharpness_val3);
	static const void Calculate_Sharpness4(IplImage *img, double &Sharpness_val4, double &detail, bool onlyDetail);
	static const void Calculate_Sharpness5(IplImage *img, vector<double>& Sharpness_val5);
	static const void Calculate_Noise1(IplImage *img,double &Noise_val1,double &avg_activity);
	static const void Calculate_Noise2(IplImage *img, vector<double>& Noise_vector2, std::string directory, std::string mosModel);
	static const void Calculate_ColorWarmth(IplImage *img, double &ColorWarmth);
	static const void Calculate_OverAndUnderExposure(IplImage *img, double& percentageOverExposure, double& percentageUnderExposure, bool dummy);
	
	//non threaded
	double Calculate_Contrast(IplImage *gimg,bool heatmap,vector<vector<double>>& totalvec);
	double Calculate_Saturation(IplImage *gimg);
	double Calculate_Exposure(IplImage *img);
	double Calculate_DynamicRange(IplImage *img);
	double Calculate_ColorWarmth(IplImage *img);
	void Calculate_OverAndUnderExposure(IplImage *img, double& percentageOverExposure, double& percentageUnderExposure);

	double Calculate_Noise1(IplImage *img,double& Noise_val1); 
	

	double Calculate_Sharpness3(IplImage *img);
	double Calculate_Sharpness4(IplImage *img, double& detail, bool onlyDetail);
	vector<double> Calculate_Sharpness5(IplImage *img);
	vector<double> VideoQuality::Calculate_NoiseInSnip(IplImage *snip, double imageResolution);
	vector<double> VideoQuality::Calculate_Noise2(IplImage *img, std::string directory, std::string mosModel);
	vector<double> VideoQuality::Calculate_Noise3(IplImage *img);
	vector<double> VideoQuality::Compute_GammaMatrix(IplImage *img, int block_size);

	//helper
	cv::Mat VideoQuality::CannyEdgeDetect(cv::Mat input_gimgMat);
	void VideoQuality::generateOutputfileNameHeatmap(string input,string attach, string &HeatMapDirecorty_Path, string& output);
	void VideoQuality::WriteToCSV(vector<double> vectorMOS, vector<double> vectorBoundedMOS, vector<double> summaryVector, vector<vector<double>>& bigVectorOfResults, vector<vector<double>>& bigVectorOfOtherResults, vector<string>& bigVectorOfFileNames);
	void VideoQuality::populate_rgamMap();
	void VideoQuality::ReadGaborFilters(int set, int scale, int orientation, int size, cv::Mat& filter);
	void VideoQuality::Calculate_EdgeStrengthUsingSobelAndCanny(cv::Mat& img, cv::Mat& cannyImg, cv::Mat& result);
	void VideoQuality::Calculate_Mean_n_Std_n_Range_of_StdforEverySubImagesOfSizeN(cv::Mat& img, int N, double& MeanOfStd, double& StdOfStd, double& RangeOfStd);
	vector<double> Mat2Vector(Mat in);
	Mat VideoQuality::Vector2Mat(vector<double> V, int r, int c);
	vector<double> ZigzagMat2Vector(Mat in);

	double SumOfVector(vector<double> inV);
	double GeoMeanofVector(vector<double> inV, int start_idx, int end_idx);

	bool CheckResultFile_Exists(const char *result_file);

	void setTotalFrameCount(int no);
	
	void CalculateAllParamValues(Mat& subimage,double& alpha,double& betaL,double& betaR);
	void Resize_image(Mat& gray_image,Mat& resized_GrayImage);
	
	
	
	bool PrintStringVectorToFile(string fileName, std::vector<string> myVector, bool putSeperators = true);
	bool PrintMatToFileNew(cv::Mat& myMat);

	
	void CreateHeatMap(Mat& imgMat,string Outputfile);
	string CreateHeatmapDirectory(string video_file);
	
	void WriteResultsToBigVectors(	FrameData& cFrame, 
									int numOfAttributesInRVector, 
									std::string& input_file,
									vector<vector<double>>& bigVectorOfResults,
									vector<vector<double>>& bigVectorOfOtherResults,
									vector<string>& bigVectorOfFileNames);

	void setResolution(double width,double height);

	string outputSharpness;			
	string outputDR;

	double MOS(std::string& category, std::string& mosModel, vector<vector<double>>& bigVectorOfResults, vector<vector<double>>& bigVectorOfOtherResults, vector<string>& bigVectorOfFileNames, std::string directory, bool onCloud);
};
