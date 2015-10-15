/*
Copyright © 2015 Intel Corporation
This program and the accompanying materials are made available under the terms of the Eclipse Public License v1.0, 
 which accompanies this distribution, and is available at http://www.eclipse.org/legal/epl-v10.html . https://github.com/viqet
* Contributors:
*    Intel Corporation - initial API and implementation and/or initial documentation
*/
#include "VideoQuality.h"
#include <hash_map>
#include <math.h>
#include <ppl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace Concurrency;


VideoQuality* VideoQuality::_instance=0;

const float VideoQuality::thresh_hold = 0.05f;
const double VideoQuality::ESP=0.0000000000000000001;

string VideoQuality::inputFile="";
double VideoQuality::resolution=0;
double VideoQuality::NF_1=0;
double VideoQuality::NF_2=0;
double VideoQuality::NF_3=0;
double VideoQuality::NF_4=0;
double VideoQuality::NF_5=0;
double VideoQuality::NF_6=0;
double VideoQuality::NF_7=0;
double VideoQuality::NF_8=0;
double VideoQuality::NF_9=0;


VideoQuality* VideoQuality::pGetInstance()
{
	if(0==_instance)
	{
		_instance=new VideoQuality();
	}
	return _instance;
}
VideoQuality::VideoQuality()
{
	frame_cnt = 0;
	capture = 0;
	first_write_to_file=TRUE;
	blockCnt=0;
	DayLight=FALSE;
	srand(time(NULL)); // to make it truely random

}

VideoQuality::~VideoQuality()
{

}


void VideoQuality::setResolution(double width,double height)
{
	resolution=(double)((ceil(width*height))/1000000);
}

FrameData* VideoQuality::getFrameData()
{ 	
	FrameData *data = new FrameData();
	xmutex.lock();
	vector<FrameData>::iterator  iter_fd;
	if(frame_data.size()!=0)
	{
		iter_fd=frame_data.begin();
		*data=*iter_fd;
		frame_data.erase(iter_fd);
		xmutex.unlock();
		return data;
	}
	else
	{
		xmutex.unlock();
		return nullptr;
	}
}

// Calculate_Sigma returns: absolute_difference[square(blur(imageChannel)) - blur(square(imageChannel))]
void Calculate_Sigma(Mat& channel,int Gaussian_window,Mat& sigma)
{
	int count=0;
	int type=channel.type();
	cv::Mat sqreGImage(channel.size().height,channel.size().width,type);	 

	cv::pow(channel,2.0,sqreGImage);

	//Gaussian filter on  image
	cv::Mat mu(channel.size().height,channel.size().width,type); //new image in Mat format

	cv::GaussianBlur( channel, mu, cv::Size( Gaussian_window,Gaussian_window ), (Gaussian_window/6.0),(Gaussian_window/6.0) ,BORDER_REPLICATE);
	//Square the image mu to calculate muSQ
	cv::Mat muSQ(channel.size().height,channel.size().width,type); // new image for squaring mu

	cv::pow(mu,2.0,muSQ); //mu squared image

	cv::Mat imagSQ(channel.size().height,channel.size().width,type); //new Mat image to store Gaussian blur of squared image
	//Gaussian filter on image square
	cv::GaussianBlur( sqreGImage, imagSQ,  cv::Size( Gaussian_window,Gaussian_window ), (Gaussian_window/6.0),(Gaussian_window/6.0) ,BORDER_REPLICATE);

	//calculate abs(muSQ,imagSQ) and store result in (diff_muSQ_imagSQ)
	cv::Mat diff_muSQ_imagSQ(channel.rows,channel.cols,type);
	cv::absdiff(imagSQ,muSQ,diff_muSQ_imagSQ);

	//create image sigma
	//sigma is calculated as sqrt (diff_muSQ_imagSQ)
	cv::Mat sigmaT(diff_muSQ_imagSQ.rows,diff_muSQ_imagSQ.cols,type);
	cv::sqrt(diff_muSQ_imagSQ,sigma);
}

// Calculate_Contrast gives mean[abs(max-min)/(max+min)], where min and max are pixel values in a grayBlock. 
double  VideoQuality::Calculate_Contrast(IplImage *img,bool heatmap,vector<vector<double>>& totalvec)
{
	int count=0;
	double contrast=0;
	double min2, max2;
	CvPoint minl, maxl;
	IplImage *gimg= cvCreateImage( cvSize( img->width, img->height ), IPL_DEPTH_8U, 1 );					 //create a new grayscale image
	cvCvtColor(img,gimg,CV_BGR2GRAY);	//convert original image to grayscale image
	vector<double> colvec;
	
	for(int r=0; r<(gimg->height);r+=size_N)
	{
		for(int c=0;c<(gimg->width);c+=size_N)
		{
			CvRect old_roi = cvGetImageROI(gimg);
			cvSetImageROI(gimg, cvRect(c,r,min(size_N,(gimg->width-c)),min(size_N,(gimg->height-r))));
			cvMinMaxLoc(gimg, &min2, &max2, &minl, &maxl);
			contrast+=((abs(max2-min2))/(ESP+(max2+min2)));
			if(heatmap) colvec.push_back((abs(max2-min2))/(ESP+(max2+min2))); 
			cvSetImageROI(gimg,old_roi);																	// reset old roi
			count++;					
		}
		if(heatmap){
			totalvec.push_back(colvec);
			colvec.clear();
		}
	}

	cvReleaseImage(&gimg);
	return (contrast/count);
}

double VideoQuality::Calculate_Saturation(IplImage *img)
{
	IplImage *gimg= cvCreateImage( cvSize( img->width, img->height ), IPL_DEPTH_8U, 3 );						 //create a new 3 channel image
	cvCvtColor(img,gimg,CV_BGR2HSV);																			//convert original image to HSV image
	CvScalar c = cvAvg(gimg);
	cvReleaseImage(&gimg);

	return c.val[1];
}


double VideoQuality::Calculate_Saturation(void *imgVoidPt)
{
	IplImage *img = (IplImage*) imgVoidPt;
	return Calculate_Saturation(img);
}


// Calculate_Exposure gives you the mean(mean(pixelsInBLock)) for all blocks where std(pixelsInBLock) > mean(std(pixelsInBLock)), in a grayImage;
double VideoQuality::Calculate_Exposure(IplImage *img)
{
	int count=0;
	double contrast=0;
	IplImage *gimg= cvCreateImage( cvSize( img->width, img->height ), IPL_DEPTH_8U, 1 );						 //create a new grayscale image
	cvCvtColor(img,gimg,CV_BGR2GRAY);																			//convert original image to grayscale image
	multimap <double,double> std_array;
	double median_val=0;
	int r =0,c=0;
	CvRect old_roi;
	CvScalar* mean=new CvScalar();
	CvScalar* std= new CvScalar();
	double std_threshold=0.0,std_sum=0.0;
	float time_frame;

	for(int r=0; r<gimg->height;r+=size_N)
	{
		for(int c=0;c<(gimg->width);c+=size_N)
		{
			old_roi = cvGetImageROI(gimg);
			cvSetImageROI(gimg, cvRect(c,r,min(size_N,(gimg->width-c)),min(size_N,(gimg->height-r))));
			cvAvgSdv(gimg,mean,std,0);																			 // calculate average Std deviation
			std_sum+=std->val[0];																				// to calculate average later- sum it up
			count++;

			//calculate mean for this block
			median_val=mean->val[0];

			cvSetImageROI(gimg,old_roi); // reset old roi
			std_array.insert(pair<double,double>(median_val, std->val[0]));
		}
	}

	std_threshold=(std_sum/(count+1));
	multimap<double,double>::iterator iter;
	int non_uni_count=0;
	double median_sum=0;
	for( iter = std_array.begin(); iter != std_array.end(); iter++ ) 
	{
		if(iter->second > std_threshold) //non-uniform area
		{
			median_sum+=iter->first;
			non_uni_count++;
		}
	}
	delete(mean);
	delete(std);
	cvReleaseImage(&gimg);
	return (median_sum/(non_uni_count+1));

}


//Calculate_DynamicRange gives you the KLDivergence of the Dist in the histofram from Uniform Dist. namely sum(p(i)log[p(i)/q(i)])
double VideoQuality::Calculate_DynamicRange(IplImage *img)
{
	int count=0;
	IplImage *gimg= cvCreateImage( cvSize( img->width, img->height ), IPL_DEPTH_8U, 1 );						//create a new grayscale image
	cvCvtColor(img,gimg,CV_BGR2GRAY);																			//convert original image to grayscale image

	int numBins = 256;
	float range[] = {0, 255};
	float *ranges[] = { range };

	CvHistogram *hist = cvCreateHist(1, &numBins, CV_HIST_ARRAY, ranges, 1);
	cvCalcHist(&gimg, hist, 0, 0);
	double SumHistValue=0.0;
	for(int i=0;i<255;i++)
	{
		double histValue = cvQueryHistValue_1D(hist, i);
		SumHistValue+=histValue;
	}
	double array_hist[256];
	double N=0.00390625;
	double KL=0.0, sum_array=0.0;

	for(int i=0;i<=255;i++)
	{
		double histValue = cvQueryHistValue_1D(hist, i);
		array_hist[i]=(histValue)/(SumHistValue+ESP);

		sum_array+=array_hist[i];
		if(array_hist[i]!=0)
		{	 
			KL+=array_hist[i]*(log((array_hist[i])/0.00390625));
		}
	}
	cvReleaseImage(&gimg);
	return KL;

}

double VideoQuality::Calculate_ColorWarmth(IplImage *img)
{
	cv::Mat img_Mat(img, true);
	cv::Mat imgMat_double;
	img_Mat.convertTo(imgMat_double, CV_64F);
	cv::Mat xyz_Mat;
	cvtColor(img_Mat, xyz_Mat, CV_RGB2XYZ);
	cv::Mat xyzMat_double;
	xyz_Mat.convertTo(xyzMat_double, CV_64F);	

	//seperate out x,y,z channels
	Mat img_x(xyz_Mat.rows, xyz_Mat.cols, CV_64F);
	Mat img_y(xyz_Mat.rows, xyz_Mat.cols, CV_64F);
	Mat img_z(xyz_Mat.rows, xyz_Mat.cols, CV_64F);
	Mat out_xyz[] = { img_x, img_y, img_z };
	int from_to_xyz[] = { 0,0, 1,1, 2,2 };
	mixChannels( &xyzMat_double, 1, out_xyz, 3, from_to_xyz, 3);	
	
	double x,y;
	double sumWarmth = 0;
	double distance;
	long count = 0;

	Mat img_x_small, img_y_small;
	cv::resize(img_x, img_x_small, cvSize(0,0), 0.25, 0.25, INTER_LINEAR);
	cv::resize(img_y, img_y_small, cvSize(0,0), 0.25, 0.25, INTER_LINEAR);

	for(int i = 0; i < img_x_small.rows; i++)
	{
		for(int j = 0; j < img_x_small.cols; j++)
		{
			x = (img_x_small.at<double>(i,j))/256.0;
			y = (img_y_small.at<double>(i,j))/256.0;

			int minDistanceIdx = -1;
			double minDistance = 9999;

			for(int idx = 0; idx < 391; idx++)
			{
				distance = pow((x - Constants::colorWarmth[idx][1]),2) + pow((y - Constants::colorWarmth[idx][2]),2);
				if(distance < minDistance)
				{
					minDistanceIdx = idx;
					minDistance = distance;
				}
			}
			sumWarmth += Constants::colorWarmth[minDistanceIdx][0];
			count++;
		}
	}

	return(sumWarmth/count);
};

double VideoQuality::Calculate_Sharpness3(IplImage *img)
{
	IplImage *gimg= cvCreateImage( cvSize( img->width, img->height ), IPL_DEPTH_8U, 1 );						 //create a new 3 channel image
	cvCvtColor(img,gimg,CV_BGR2GRAY);		
	IplImage* out = cvCreateImage(cvSize(gimg->width, gimg->height),IPL_DEPTH_16S,1);
	cvLaplace(gimg, out, 1);		

	double minVal; 
	double maxVal; 
	Point minLoc; 
	Point maxLoc;

	minMaxLoc( Mat(out, true), &minVal, &maxVal, &minLoc, &maxLoc );
	double maxLap = maxVal;
	cvReleaseImage(&out);
	return (double)maxLap;
}

cv::Mat VideoQuality::CannyEdgeDetect(cv::Mat input_gimgMat)
{
		int lower_canny_threshold=40, upper_canny_threshold=80;
        cv::Mat gauss_filtered_im, gauss_filtered_im_int, contours;
        float gaussian_sigma_x=2, gaussian_sigma_y=2;
        cv::GaussianBlur( input_gimgMat, gauss_filtered_im,  cv::Size(6* gaussian_sigma_x+1,6* gaussian_sigma_y+1), gaussian_sigma_x, gaussian_sigma_y, BORDER_REFLECT);   
		gauss_filtered_im.convertTo(gauss_filtered_im_int, CV_8U);
		cv::Canny(gauss_filtered_im_int, contours, lower_canny_threshold, upper_canny_threshold);    
        contours = contours/255; 
	    return contours;

}

double VideoQuality::Calculate_Sharpness4(IplImage *img, double& detail, bool onlyDetail)
{
	IplImage *gimg= cvCreateImage( cvSize( img->width, img->height ), IPL_DEPTH_8U, 1); 
	cvCvtColor(img,gimg,CV_BGR2GRAY);

	cv::Mat gimgMat(gimg,true);	
	cv::Mat blurredOnce_gimgMat, blurredTwice_gimgMat;		

	cv::Mat gimgMat_double; 
	gimgMat.convertTo(gimgMat_double, CV_64F);
	cv::GaussianBlur( gimgMat_double,      blurredOnce_gimgMat,  cv::Size( 7,7 ), (7.0/6.0), (7.0/6.0), BORDER_CONSTANT); 
	cv::GaussianBlur( blurredOnce_gimgMat, blurredTwice_gimgMat, cv::Size( 7,7 ), (7.0/6.0), (7.0/6.0), BORDER_CONSTANT);
	cv::Mat contours = CannyEdgeDetect(gimgMat_double);

	cv::Scalar contours_count = cv::sum(contours);
	detail = (double) contours_count.val[0];

	if(onlyDetail) 
		return 0.0;

	cv::Mat contours_double;
	contours.convertTo(contours_double, CV_64F);

	cv::Mat ED1, ED2;  
	VideoQuality::Calculate_EdgeStrengthUsingSobelAndCanny(blurredTwice_gimgMat, contours_double, ED1);
	cv::Scalar sharpness = cv::sum(ED1);                         // <------- ALGO 2
	double sharpness_d = (double) sharpness.val[0];		

	return sharpness_d; 	
}

void VideoQuality::Calculate_EdgeStrengthUsingSobelAndCanny(cv::Mat& img, cv::Mat& cannyImg, cv::Mat& result)
{
	cv::Mat sobelH_OnImg;
	cv::Mat sobelV_OnImg;

	cv::Sobel(img, sobelH_OnImg, -1, 0, 1, 3, 1, 0, BORDER_CONSTANT);
	cv::Sobel(img, sobelV_OnImg, -1, 1, 0, 3, 1, 0, BORDER_CONSTANT);
	cv::Mat abs_sobelH_OnImg = cv::abs(sobelH_OnImg*(-1)); 
	cv::Mat abs_sobelV_OnImg = cv::abs(sobelV_OnImg*(-1)); 

	result = ((abs_sobelH_OnImg + abs_sobelV_OnImg)*0.5).mul(cannyImg);	
}

void VideoQuality::ReadGaborFilters(int set, int scale, int orientation, int size, cv::Mat& filter)
{
	string line; 
	int line_number = 0;
	double x[100][100] ; // large enough space

	char set_str[4], scale_str[4], ori_str[4];
	sprintf(  set_str, "%d", set);
	sprintf(scale_str, "%d", scale);
	sprintf(  ori_str, "%d", orientation);
	string filename = string(".\\GaborFiltersInText\\Set") + set_str + "\\gabor_scale" + scale_str +"_ori" + ori_str + ".txt";	

	// looks in $PROJECT\vgui\bin\x64\release
	ifstream myfile_r (filename); 
	if (myfile_r.is_open())
	{
		while ( std::getline(myfile_r,line) )
		{
			std::istringstream in(line);	

			for(int col = 0; col < size; col++) 
			{	in >> x[line_number][col]; }

			line_number++;
		}
		myfile_r.close();
	}

	Mat dataMat(100, 100, CV_64F, x);
	Mat roi(dataMat, Rect(0, 0, size, size));
	filter = (roi)*1.0; // just 'filter = roi' would overwrite older filter reads
}

vector<double> VideoQuality::Calculate_Sharpness5(IplImage *img)
{
	IplImage *gimg= cvCreateImage( cvSize( img->width, img->height ), IPL_DEPTH_8U, 1 );						 //create a new 3 channel image	
	cvCvtColor(img,gimg,CV_BGR2GRAY);	 

	cv::Mat gimgMat(gimg,true); 

	cv::Mat contours, contours_double;  
	contours = CannyEdgeDetect(gimgMat);
	contours.convertTo(contours_double, CV_64F);

	cv::Mat gimgMat_double;
	gimgMat.convertTo(gimgMat_double, CV_64F);	

	vector<double> resultSharpness;	//	<------- ALGO 3

	int Size[9] = { 49, 25, 17, 15, 11, 11, 9, 9, 7}; // defining the 9 sizes;

	for(int idx = 0; idx < 9; idx++)
	{
		cv::Mat result;
		cv::Mat cgbL(gimgMat_double.rows, gimgMat_double.cols,  CV_64F, 0.00);

		for(int orient = 0; orient < 4; orient++)
		{
			int size = Size[idx];
			cv::Mat kernel;

			ReadGaborFilters(1, (idx+1), (orient+1), size, kernel); 			

			cv::filter2D(gimgMat_double, result, -1, kernel, cv::Point(-1,-1), 0.0, BORDER_CONSTANT);			
			cgbL += result*0.25;

		}		

		cv::Mat EDgab = contours_double.mul(cgbL);
		cv::Mat EDgab_smaller = EDgab(cv::Rect(10, 10, EDgab.size().width -20 , EDgab.size().height -20));
		cv::Scalar sharpness = cv::sum(cv::abs(EDgab_smaller));   		
		resultSharpness.push_back(((double) sharpness.val[0]));					// <------- ALGO 3
	}

	return resultSharpness;
}

vector<double> VideoQuality::Calculate_NoiseInSnip(IplImage *snip, double imageResolution)
{
	vector<double> resultNoiseInSnip;

	IplImage *snip_gray= cvCreateImage( cvSize( snip->width, snip->height ), IPL_DEPTH_8U, 1 );	
	cvCvtColor(snip,snip_gray,CV_BGR2GRAY);	

	cv::Mat snip_Mat(snip, true);
	cv::Mat snip_grayMat(snip_gray, true);

	cv::Mat snip_grayMat_double;
	snip_grayMat.convertTo(snip_grayMat_double, CV_64F);

	int sc = 3;

	cv::Mat currentImg, nextImg;
	currentImg = snip_grayMat_double;

	double mean_std_std_gray  = 0.0;
	double mean_mean_std_gray = 0.0;

	for(int i = 0; i < 3; i++)
	{
		cv::resize(currentImg, nextImg, cvSize(0,0), 0.5, 0.5, INTER_LINEAR);

		double mean, std, range;
		Calculate_Mean_n_Std_n_Range_of_StdforEverySubImagesOfSizeN(currentImg, sc, mean, std, range);
		mean_std_std_gray  +=  std/3.0;
		mean_mean_std_gray += mean/3.0;
		currentImg = nextImg;
	}

	//build an image pyramid for hue snip
	cv::Mat snip_Mat_double;
	snip_Mat.convertTo(snip_Mat_double, CV_64F);

	cv::Mat hsv_snipMat;
	cvtColor(snip_Mat, hsv_snipMat, CV_BGR2HSV);
	cv::Mat hsv_snipMat_double;
	hsv_snipMat.convertTo(hsv_snipMat_double, CV_64F);

	//seperate out h,s,v channels
	Mat snip_h(hsv_snipMat_double.rows, hsv_snipMat_double.cols, CV_64F);
	Mat snip_s(hsv_snipMat_double.rows, hsv_snipMat_double.cols, CV_64F);
	Mat snip_v(hsv_snipMat_double.rows, hsv_snipMat_double.cols, CV_64F);
	Mat out_hsv[] = { snip_h, snip_s, snip_v };
	int from_to_hsv[] = { 0,0, 1,1, 2,2 };
	mixChannels( &hsv_snipMat_double, 1, out_hsv, 3, from_to_hsv, 3);

	snip_h = snip_h.mul((double)(1/180.0)); 				

	//seperate out r,g,b channels
	Mat snip_r(snip_Mat_double.rows, snip_Mat_double.cols, CV_64F);
	Mat snip_g(snip_Mat_double.rows, snip_Mat_double.cols, CV_64F);
	Mat snip_b(snip_Mat_double.rows, snip_Mat_double.cols, CV_64F);
	Mat out[] = { snip_b, snip_g, snip_r };
	int from_to[] = { 0,0, 1,1, 2,2 };
	mixChannels( &snip_Mat_double, 1, out, 3, from_to, 3);

	Scalar meanR = cv::mean(snip_r);
	Scalar meanG = cv::mean(snip_g);
	Scalar meanB = cv::mean(snip_b);

	cv::Mat CRi = (snip_r).mul(1/((double)meanR.val[0] + 1)); 
	cv::Mat CGi = (snip_g).mul(1/((double)meanG.val[0] + 1));
	cv::Mat CBi = (snip_b).mul(1/((double)meanB.val[0] + 1));

	Mat rgb_std_im(snip_Mat_double.rows, snip_Mat_double.cols, CV_64F);
	Mat temp(3, 1, CV_64F);

	for(int i = 0; i < snip_Mat_double.rows; i++)
	{
		for(int j = 0; j < snip_Mat_double.cols; j++)
		{
			Scalar std, mean;

			temp.at<double>(0) = CRi.at<double>(i,j);
			temp.at<double>(1) = CGi.at<double>(i,j);
			temp.at<double>(2) = CBi.at<double>(i,j);

			cv::meanStdDev(temp, mean, std);
			rgb_std_im.at<double>(i,j) = std.val[0]* pow(1.5, 0.5); 
		}
	} 

	double color_noise2 = mean(rgb_std_im).val[0]; 

	Scalar std_hsv, mean_hsv;
	cv::meanStdDev(snip_h, mean_hsv, std_hsv);
	double color_noise3 = std_hsv.val[0]; 			

	double geoMean_temp = pow(color_noise2 * color_noise3, 0.5);
	double resolutionFactor = imageResolution / 1000000.0; // Get the N from N MegaPixels !

	double score = pow(geoMean_temp * mean_std_std_gray * (mean_mean_std_gray/resolutionFactor), 1.0/3.0);

	resultNoiseInSnip.push_back(mean_mean_std_gray);
	resultNoiseInSnip.push_back(mean_std_std_gray);
	resultNoiseInSnip.push_back(score);

	return resultNoiseInSnip;
}

void VideoQuality::Calculate_Mean_n_Std_n_Range_of_StdforEverySubImagesOfSizeN(cv::Mat& img, int N, double& MeanOfStd, double& StdOfStd, double& RangeOfStd)
{
	vector<double> v_stdDev;
	Scalar std, mean;	

	
	double opencv2matlabFactorForStd = pow(((N*N)/(double)(N*N - 1)), 0.5);  

	int count = 0;
	for(int i =0; i < img.cols -N +1 ; i++)
	{
		for(int j =0; j < img.rows -N +1; j++)
		{
			Mat roi(img, Rect(i, j, N, N)); 

			cv::meanStdDev(roi, mean, std);
			v_stdDev.push_back(opencv2matlabFactorForStd * (double)std.val[0]);		

			count++;
		}
	}
	Mat vector2Mat(v_stdDev,true); 
	cv::meanStdDev(vector2Mat, mean, std);	

	
	double opencv2matlabFactor2ForStd = pow(((count)/(double)(count - 1)), 0.5);

	MeanOfStd = mean.val[0];
	StdOfStd = opencv2matlabFactor2ForStd * std.val[0];

	cv::sort(vector2Mat, vector2Mat, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
	RangeOfStd = vector2Mat.at<double>(count -1,0) - vector2Mat.at<double>(0,0); 

	int debug = 0;
}


void VideoQuality::Calculate_OverAndUnderExposure(IplImage *img, double& percentageOverExposure, double& percentageUnderExposure)
{
	//create a new grayscale image
	IplImage *gimg= cvCreateImage( cvSize( img->width, img->height ), IPL_DEPTH_8U, 1 );						
	cvCvtColor(img,gimg,CV_BGR2GRAY);
	cv::Mat gimgMat(gimg,true);
	cv::Mat gimgMat_double;
	gimgMat.convertTo(gimgMat_double, CV_64F);

	int filter_size = ceil(min(gimgMat_double.rows,gimgMat_double.cols)/8); 
	double filterCellValue = 1.0/(filter_size*filter_size); 

	cv::Mat averagingFilter(filter_size, filter_size, CV_64F, Scalar(filterCellValue));
	cv::Mat windowAveraged;

	
	cv::filter2D(gimgMat_double, windowAveraged, -1, averagingFilter, cv::Point(-1,-1), 0.0, BORDER_REFLECT);

	double totalPixelCount = windowAveraged.rows * windowAveraged.cols;
	double totalUnderExposed = 0.0;
	double totalOverExposed = 0.0;

	double pixelValue;
	for(int row=0; row < windowAveraged.rows; row++)
	{
		for(int col=0; col < windowAveraged.cols; col++)
		{
			pixelValue = windowAveraged.at<double>(row,col);

			if (pixelValue > 254) totalOverExposed += 1.0;
			if (pixelValue < 2) totalUnderExposed += 1.0;
		}
	}

     percentageOverExposure  = (totalOverExposed/totalPixelCount)*100.0;
	 percentageUnderExposure = (totalUnderExposed/totalPixelCount)*100.0;
}

vector<double> VideoQuality::Mat2Vector(Mat in)
{
	vector<double> out;
    for (int i=0; i < in.rows; i++) 
	{
		for (int j =0; j < in.cols; j++)
		{
			out.push_back(in.at<double>(i,j));
		}
	}
	return out;

}

Mat VideoQuality::Vector2Mat(vector<double> V, int r, int c)
{
	Mat M2=Mat(r,c,CV_64F);
	memcpy(M2.data,V.data(),V.size()*sizeof(double));

	return M2;
}

vector<double> VideoQuality::ZigzagMat2Vector(Mat in)
{
	vector<double> out;

	int v = 0,				h = 0;
	int vmin = 0,			hmin = 0;
	int vmax = in.rows -1 , hmax = in.cols -1;	

	while( v <= vmax && h <= hmax)
	{
		if( (h+v)%2 == 0)								// going up
		{
			if(v == vmin)								// if we got to the first line
			{	
				out.push_back(in.at<double>(v,h));

				if(h == hmax)   v++;
				else				h++;
			} else if(h == hmax && v < vmax)			// if we got to the last column
			{
				out.push_back(in.at<double>(v,h));
				v++;

			} else if( v > vmin && h < hmax)			// all other cases
			{
				out.push_back(in.at<double>(v,h));
				v--;
				h++;
			}
		} else											// going down
		{
			if( v == vmax && h <= hmax)					// if we got to the last line
			{
				out.push_back(in.at<double>(v,h));
				h++;
			} else if(h == hmin)						// if we got to the first column
			{
				out.push_back(in.at<double>(v,h));

				if(v == vmax) h++;
				else          v++;
			} else if ( v < vmax & h > hmin)			// all other cases
			{
				out.push_back(in.at<double>(v,h));
				v++;
				h--;
			}
		}

		if( v == vmax && h == hmax )					// bottom right elements
		{
			out.push_back(in.at<double>(v,h));
			break;
		}
	}

	return out;
}

double VideoQuality::SumOfVector(vector<double> inV)
{
	double sum = 0.0;
	for(int i =0; i < inV.size(); i++)
		sum += inV[i];
	return sum;
}

double VideoQuality::GeoMeanofVector(vector<double> inV, int start_idx, int end_idx)
{
	if(start_idx > end_idx) return 0;

	double product = 1.0;
	for(int i =start_idx; (i < inV.size()) && (i <= end_idx) && (i>=0); i++)
		product *= inV[i];

	return std::pow(product, (1.0/(end_idx - start_idx + 1)));	
}

class double_with_idx
{
	public:

	double value;
	int idx;
	double_with_idx(double _value, int _idx)
	{
		value = _value;
		idx = _idx;
	}		
};

bool sort_double_with_idx(double_with_idx v1, double_with_idx v2)
{
	return (v1.value < v2.value);
}

vector<double> VideoQuality::Compute_GammaMatrix(IplImage *img, int block_size)
{
	IplImage *gimg= cvCreateImage( cvSize( img->width, img->height ), IPL_DEPTH_8U, 1 );						
	cvCvtColor(img, gimg, CV_BGR2GRAY);
	cv::Mat gimgMat(gimg, true);

	cv::Mat gimgMat_double;
	gimgMat.convertTo(gimgMat_double, CV_64F);	

	vector<vector<double>> dct5x5;
	for(int rr = 0; rr <= gimgMat_double.rows -block_size; rr += block_size)
	{
		for(int cc = 0; cc <= gimgMat_double.cols -block_size; cc += block_size)
		{
			cv::Mat snip(gimgMat_double, Rect(cc, rr, block_size, block_size));
			
			cv::Mat dct_features;
			dct(snip,dct_features);

			vector<double> dct_vector = Mat2Vector(dct_features); 
			dct5x5.push_back(dct_vector);
		}
	}

	vector<double> g,r;
	for(double val = 0.03; val <= 10 ; val+=0.001)
	{
		g.push_back(val);
		r.push_back( Gamma(1.0/val) * Gamma(3.0/val) / (pow(Gamma(2.0/val),2)));
	}

	vector<double> gama_freq, rho_freq, var_freq, mean_freq;

	for(int i=0; i < block_size*block_size; i++)
	{
		vector<double> temp;
		for(int j=0; j<dct5x5.size(); j++)
		{
			temp.push_back(dct5x5[j][i]);
		}

		Mat temp_mat = Vector2Mat(temp, dct5x5.size(), 1);

		cv::Scalar mean_scalar, std_gauss;
		cv::meanStdDev(temp_mat, mean_scalar, std_gauss);

		double mean_gauss = mean_scalar.val[0];
		double var_gauss = std_gauss.val[0] * std_gauss.val[0];

		double mean_abs = pow(mean(abs(temp_mat - mean_gauss)).val[0],2);
		double rho = var_gauss/(mean_abs + 0.0000001);

		double gamma_gauss = 11;

		for(int x = 0; x < g.size() - 1; x++)
		{
			if(rho <= r[x] && rho > r[x+1])
			{
				gamma_gauss = g[x];
				break;
			}
		}

		gama_freq.push_back(gamma_gauss);
		rho_freq.push_back(rho);
		var_freq.push_back(var_gauss);
		mean_freq.push_back(mean(abs(temp_mat)).val[0]);
	}

	Mat gama_matrix = Vector2Mat(gama_freq, block_size, block_size);
	Mat rho_matrix = Vector2Mat(rho_freq, block_size, block_size);
	Mat var_matrix = Vector2Mat(var_freq, block_size, block_size);
	Mat mean_matrix = Vector2Mat(mean_freq, block_size, block_size);

	vector<vector<double>> zigZags;
	zigZags.push_back(ZigzagMat2Vector(gama_matrix));
	zigZags.push_back(ZigzagMat2Vector( rho_matrix));
	zigZags.push_back(ZigzagMat2Vector( var_matrix));
	zigZags.push_back(ZigzagMat2Vector(mean_matrix));

	vector<double> _return;

	int binCount = 6;
	for(int i =0; i < zigZags.size(); i++)
	{
		vector<double> temp = zigZags[i];
		int size = temp.size();

		for(int j = 0; j < size; j+= size/binCount)
		{
			vector<double> subTemp = vector<double>(temp.begin() + (j==0?1:j) /*to remove DC*/ , temp.begin() + j + size/binCount);
			_return.push_back(SumOfVector(subTemp)/subTemp.size());
		}
	}	
	return _return;
}

vector<double> VideoQuality::Calculate_Noise3(IplImage *img)
{
	double resize_factor = 0.75;
	IplImage *img_resized = cvCreateImage( cvSize(resize_factor * img->width, resize_factor * img->height), img->depth, img->nChannels);
	cvResize(img, img_resized);

	IplImage *gimg= cvCreateImage( cvSize( img_resized->width, img_resized->height ), IPL_DEPTH_8U, 1 );						
	cvCvtColor(img_resized, gimg, CV_BGR2GRAY);
	cv::Mat gimgMat(gimg, true);

	cv::Mat gimgMat_double;
	gimgMat.convertTo(gimgMat_double, CV_64F);
	 
	// get all Gabor filters here !
	// outer vector for orient and the inner for size;
	int Size[4] = { 45, 25, 13, 7};
	vector<vector<cv::Mat>> gaborFilters;
	for(int i = 0 ; i < 4 ; i++)
	{
		vector<cv::Mat> innerVector;
		gaborFilters.push_back(innerVector);
	}
	for(int orient = 0; orient < 4; orient++)		
	{
		for(int idx = 0; idx < 4; idx++)
		{
			int size = Size[idx];
			cv::Mat kernel;
			ReadGaborFilters( 3, (idx+1), (orient+1), size, kernel); 
			gaborFilters[orient].push_back(kernel); 
		}
	}

	int block_size = 80;
	vector<vector<double>> features;

	for(int rr = 0; rr <= gimgMat_double.rows -block_size; rr += block_size)
	{
		for(int cc = 0; cc <= gimgMat_double.cols -block_size; cc += block_size)
		{
			cv::Mat snip(gimgMat_double, Rect(cc, rr, block_size, block_size));

			double mean_std, std_std, range_std;
			Calculate_Mean_n_Std_n_Range_of_StdforEverySubImagesOfSizeN(snip, 20, mean_std, std_std, range_std);
				

			cv::Mat cgb;
			int count = 0;
			double sum_score = 0.0, sum_ratio = 0.0, sum_mean_cgb = 0.0, sum_renyi_cgb = 0.0;

			for(int orient = 0; orient < 4; orient++)				
			{	
				for(int idx = 0; idx < 4; idx++)					
				{
					count++;
					
					cv::filter2D(snip, cgb, -1, gaborFilters[orient][idx], cv::Point(-1,-1), 0.0, BORDER_REFLECT);

					cv::Mat temp(cgb.size().height, cgb.size().width, cgb.type());
					cv::pow(cgb, 2.0, temp);


					double sum_squared_cgb = cv::sum(temp).val[0];
					temp = temp/sum_squared_cgb;

					cv::Mat temp1(cgb.size().height, cgb.size().width, cgb.type());
					cv::pow(temp, 3.0, temp1);
					double sum_raisedto3_temp = cv::sum(temp1).val[0];

					double renyi_cgb = (-0.5)*log(sum_raisedto3_temp);				

					double mean_cgb = mean(cv::abs(cgb)).val[0];

					vector<double> snip_vector = Mat2Vector(snip);
					std::sort(snip_vector.begin(),snip_vector.end());
					double snip_ratio = snip_vector[0.75 * snip_vector.size()] / snip_vector[0.25 * snip_vector.size()];

					snip_vector.size();
					
					sum_renyi_cgb  += renyi_cgb;
					sum_mean_cgb   += mean_cgb;
					sum_score	   += (mean_cgb/renyi_cgb);
					sum_ratio	   += snip_ratio;											
				}		
			}
			
			double score			= sum_score/count; 
			double ratio			= sum_ratio/count; 
			double mean_mean_cgb	= sum_mean_cgb/count; 
			double mean_renyi_cgb	= sum_renyi_cgb/count;

			vector<double> snip_feature;

			cv::Mat dct_features;
			dct(snip,dct_features);
			cv::Mat abs_dct_features = cv::abs(dct_features);	
			vector<double> longVector = VideoQuality::ZigzagMat2Vector(abs_dct_features);
			vector<double> cornerVector = vector<double>(longVector.begin() + 4500, longVector.end());

			double dct_feat = SumOfVector(cornerVector)/cornerVector.size();					

			snip_feature.push_back(dct_feat);
			snip_feature.push_back(mean_std);
			snip_feature.push_back(std_std);
			snip_feature.push_back(score);
			snip_feature.push_back(ratio);
			snip_feature.push_back(mean_mean_cgb);
			snip_feature.push_back(mean_renyi_cgb);

			features.push_back(snip_feature);
		}
	}

	vector<double_with_idx> temp_all_snips;
	vector<double> noise_all_snips;

	for(int i =0; i < features.size(); i++)
	{
		double temp_geomean = GeoMeanofVector(features[i], 1, features[i].size() - 1);
		double_with_idx temp(temp_geomean,i);
		temp_all_snips.push_back(temp);

		noise_all_snips.push_back(GeoMeanofVector(features[i], 1, 2));
	}

	std::sort(temp_all_snips.begin(), temp_all_snips.end(), sort_double_with_idx);

	double sum_noise_new = 0.0;
	double sum_freq_noise = 0.0;
	int count_snips = 0;
	for(int i =0; i < 0.01 * temp_all_snips.size(); i++) 
	{
		count_snips++;
		sum_noise_new += noise_all_snips[temp_all_snips[i].idx];
		sum_freq_noise += features[temp_all_snips[i].idx].at(0);
	}

	double noise_new = sum_noise_new/count_snips;
	double freq_noise = sum_freq_noise/count_snips;

	vector<double> noise_vector;
	noise_vector.push_back(noise_new);
	noise_vector.push_back(freq_noise);
	return(noise_vector);
}

vector<double> VideoQuality::Calculate_Noise2(IplImage *img, std::string directory, std::string mosModel)
{
	const int resultLength = 3;

	//create a new grayscale image
	IplImage *gimg= cvCreateImage( cvSize( img->width, img->height ), IPL_DEPTH_8U, 1 );						
	cvCvtColor(img,gimg,CV_BGR2GRAY);
	cv::Mat gimgMat(gimg,true);
	cv::Mat imgMat(img, true);

	cv::Mat gimgMat_double;
	gimgMat.convertTo(gimgMat_double, CV_64F);

	int Size[4] = { 45, 25, 13, 7}; 

	int block_size = ceil(min(gimgMat_double.rows,gimgMat_double.cols)/8);

	// initialize the vector of vector
	vector<vector<double>> allResult;
	for(int i = 0; i < resultLength; i++)
	{
		vector<double> temp;
		allResult.push_back(temp);
	}

	vector<vector<double>> testDataAll;
	vector<int> rr_vec;
	vector<int> cc_vec;

	// get all Gabor filters here !
	// outer vector for orient and the inner for size;
	vector<vector<cv::Mat>> gaborFilters;
	for(int i = 0 ; i < 4 ; i++)
	{
		vector<cv::Mat> innerVector;
		gaborFilters.push_back(innerVector);
	}
	for(int orient = 0; orient < 4; orient++)		
	{
		for(int idx = 0; idx < 4; idx++)
		{
			int size = Size[idx];
			cv::Mat kernel;
			ReadGaborFilters( 3, (idx+1), (orient+1), size, kernel); 
			gaborFilters[orient].push_back(kernel);
		}
	}


	for(int rr = 0; rr <= gimgMat_double.rows -block_size; rr += block_size)
	{
		for(int cc = 0; cc <= gimgMat_double.cols -block_size; cc += block_size)
		{
			cv::Mat block(gimgMat_double, Rect(cc, rr, block_size, block_size));

			vector<double> testData;	

			for(int orient = 0; orient < 4; orient++)
				
			{				
				cv::Mat cgbL(block_size, block_size,  CV_64F, 0.00);
				cv::Mat gresult; 

				for(int idx = 0; idx < 4; idx++)
					
				{
					int size = Size[idx];
					
					cv::filter2D(block, gresult, -1, gaborFilters[orient][idx], cv::Point(-1,-1), 0.0, BORDER_REFLECT);//BORDER_CONSTANT);
					cgbL += gresult/4;					
				}

				double alpha0,betaL0,betaR0;			
				CalculateAllParamValues(cgbL, alpha0, betaL0, betaR0);

				cv::Scalar mean, std;
				cv::meanStdDev(cv::abs(cgbL), mean, std);

				double mean_d = mean.val[0];
				double var_d = std.val[0] * std.val[0];

				testData.push_back((double)mean_d);
				testData.push_back((double)var_d);
				testData.push_back((double)alpha0);
				testData.push_back((double)betaL0);
				testData.push_back((double)betaR0);				
			}

			testDataAll.push_back(testData);
			rr_vec.push_back(rr);
			cc_vec.push_back(cc);

		}
	}	

	std::vector<double> predictedLabels;
	std::vector<std::vector<double>> classProbability;
	vector<double> predictedLabel;
	vector<double> finalResult;
	
	if(mosModel.compare(Constants::GenericMOS_LibSvm)== 0 || mosModel.compare(Constants::CategoryMOS_LibSvm)== 0)
	{

		//testing libSVM start
		string model_file = "flatRegionModelData.txt";
	

			vector<svm_node *> testDataSVM;
		int max_nr_attr = testDataAll[0].size(); // length of attribute string
		for(int i=0; i<testDataAll.size();i++)
		{
			struct svm_node *x = (struct svm_node *) malloc((max_nr_attr+1)*sizeof(struct svm_node));
			int j=0;
			for(; j < testDataAll[i].size(); j++)
			{
				x[j].index = (j+1);
				x[j].value = testDataAll[i][j];
			}
			x[j].index = -1; // termination indicator for libSVM
			testDataSVM.push_back(x);
		}
	
		struct svm_model* input_model;
		if((input_model=svm_load_model(model_file.c_str()))==0)
		{	
			fprintf(stderr,"can't open model file %s\n", model_file.c_str());			
			for(int i = 0; i < resultLength; i++)
			{
				finalResult.push_back((double)(-3.142)); 
			}
			return finalResult; 
		}
		
		SVMClass svmTestObject;
		svmTestObject.main_svmPredictFileless(1, testDataSVM, input_model, predictedLabels, classProbability); //classProbability[i][1] has needed probability !
	
		
	}

	for(int i =0; i < testDataAll.size(); i++)
	{

		double oddsOfBeingFlat = 0.0;
		if(mosModel.compare(Constants::GenericMOS_LibSvm)== 0 || mosModel.compare(Constants::CategoryMOS_LibSvm)== 0)
		{
			oddsOfBeingFlat = classProbability[i][1];
		}

		if(oddsOfBeingFlat > 0.9)
		{
			int cc = cc_vec[i];
			int rr = rr_vec[i];

			CvRect old_roi = cvGetImageROI(img);
			cvSetImageROI(img, Rect(cc, rr, block_size, block_size));
			IplImage* cropped = cvCreateImage( cvSize( block_size, block_size), img->depth, img->nChannels );  
			cvCopy( img, cropped );
			cvSetImageROI(img, old_roi);

			vector<double> noiseValue = Calculate_NoiseInSnip(cropped, ((double)gimgMat_double.rows) * ((double)gimgMat_double.cols));

			//Filter out overExposed or underExposed blocks
			bool flagOverOrUnderExposed = false;
			if(noiseValue[2] == 0.0) flagOverOrUnderExposed = true;	//Since noiseValue[2] is the complex noise value that also gets displayed on the screen!
			if(!flagOverOrUnderExposed)
			{
				for(int i = 0; i < resultLength; i++)
				{
					allResult[i].push_back(noiseValue[i]);
				}
			}


		}
	}

	const double percentile = 0.50; // 50% percentile
	if( allResult[0].size() > 0 ) // to check if algo has identified any flat regions
	{
		// if 6 elements, pick pos 2 and 3;
		// if 5 elements, pick pos 2 (assuming first pos is '0'

		double percentilePosn = ((double)(allResult[0].size()-1)*percentile);

		int percentilePosn1 = 0, percentilePosn2 = 0;

		if( percentilePosn > ((int)(percentilePosn)))
		{
			percentilePosn1 = (int)(percentilePosn);
			percentilePosn2 = (int)(percentilePosn) + 1;
		}
		else {
			percentilePosn1 = (int)(percentilePosn);
			percentilePosn2 = (int)(percentilePosn);
		}

		for(int i = 0; i < resultLength; i++)
		{
			Mat vector2Mat(allResult[i],true); //convert vector to Mat
			cv::sort(vector2Mat, vector2Mat, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);

			double valueUsed = (vector2Mat.at<double>(percentilePosn1) + vector2Mat.at<double>(percentilePosn2))/2;
			finalResult.push_back(valueUsed);
		}
	}
	else
	{
		for(int i = 0; i < resultLength; i++)
		{
			finalResult.push_back((double)(-3.142)); 
		}
	}	

	return finalResult;	
}


double VideoQuality::Calculate_Noise1(IplImage *img,double& Noise_val1)
{
	Mat input(img);
	Mat input_32(input.size().height,input.size().width,CV_64F);
	input.convertTo(input_32,CV_64F);		//convert to 32 bit

	Mat channel[3];

	
	split(input_32, channel);
	int Gaussian_window=ceil(channel[0].rows/120.0);
	if(Gaussian_window%2 ==0)
	{
		Gaussian_window=Gaussian_window+1;
	}
	Mat sigmaRed,sigmaBlue,sigmaGreen;
	Calculate_Sigma(channel[0],Gaussian_window,sigmaBlue);
	Calculate_Sigma(channel[1],Gaussian_window,sigmaGreen);
	Calculate_Sigma(channel[2],Gaussian_window,sigmaRed);


	Mat AverageSigma=(sigmaRed+sigmaBlue+sigmaGreen)/3;
	Scalar  meanv=mean(AverageSigma);
	double noiseavg= meanv.val[0];
	vector <double> pixelVector;
	for(int i = 0; i < AverageSigma.cols; i++)
	{
		for(int j = 0; j < AverageSigma.rows; j ++)
		{
			double val= AverageSigma.at<double>(j,i);
			if(val>0.00001)
			{
				pixelVector.push_back(AverageSigma.at<double>(j,i));
			}
		}
	}

	std::sort(pixelVector.begin(),pixelVector.end());

	Mat pixelVectorToMat(pixelVector,true);

	meanv=mean(pixelVectorToMat);
	double avg_activity= meanv.val[0];

	int  f=floor(0.005*pixelVector.size());
	vector <double> NewpixelVector;

	for(int i=0;i<f;i++)
	{
		NewpixelVector.push_back(pixelVector.at(i));
	}
	Mat NewpixelVectorToMat(NewpixelVector,true);
	Scalar  meanv2=mean(NewpixelVectorToMat);

	Noise_val1= meanv2.val[0];

	return avg_activity;

}

/////////////////////////////////thread methods/////////////////////////
const void  VideoQuality::Calculate_Contrast(IplImage *img,double &Contrast_val)
{
	VideoQuality* vq= pGetInstance();
	vector<vector<double>> totalvec;
	Contrast_val= vq->Calculate_Contrast(img,0,totalvec);

}

const void  VideoQuality::Calculate_Contrast(void *imgVoidPt,double &Contrast_val)
{
	IplImage *img = (IplImage*) imgVoidPt;
	Calculate_Contrast(img, Contrast_val);
}

const void VideoQuality::Calculate_Saturation(IplImage *img, double &Saturation_val)
{
	VideoQuality* vq= pGetInstance();
	Saturation_val= vq->Calculate_Saturation(img);
}

const void VideoQuality::Calculate_Exposure(IplImage *img,double &Exposure_val)
{
	VideoQuality* vq= pGetInstance();
	Exposure_val= vq->Calculate_Exposure(img);
}



const void VideoQuality::Calculate_DynamicRange(IplImage *img,double &DynamicRange_val)
{
	VideoQuality* vq= pGetInstance();
	DynamicRange_val= vq->Calculate_DynamicRange(img);	
}


const void  VideoQuality::Calculate_Sharpness3(IplImage *img,double &Sharpness_val3)
{
	VideoQuality* vq= pGetInstance();
	Sharpness_val3= vq->Calculate_Sharpness3(img);
}

const void  VideoQuality::Calculate_Sharpness4(IplImage *img,double &Sharpness_val4, double &detail, bool onlyDetail)
{
	VideoQuality* vq= pGetInstance();
	Sharpness_val4= vq->Calculate_Sharpness4(img, detail, onlyDetail);
}

const void VideoQuality::Calculate_ColorWarmth(IplImage *img, double &ColorWarmth)
{
	VideoQuality* vq= pGetInstance();
	ColorWarmth = vq->Calculate_ColorWarmth(img);
}

const void  VideoQuality::Calculate_OverAndUnderExposure(IplImage *img, double& percentageOverExposure, double& percentageUnderExposure, bool dummy)
{
	VideoQuality* vq= pGetInstance();
	vq->Calculate_OverAndUnderExposure(img, percentageOverExposure, percentageUnderExposure);
}

const void VideoQuality::Calculate_Sharpness5(IplImage *img, vector<double>& Sharpness_val5)
{
	VideoQuality* vq= pGetInstance();
	Sharpness_val5 = vq->Calculate_Sharpness5(img);
}

const void VideoQuality::Calculate_Noise2(IplImage *img, vector<double>& Noise_vector2, std::string directory, std::string mosModel)
{
	VideoQuality* vq= pGetInstance();
	Noise_vector2 = vq->Calculate_Noise2(img, directory, mosModel);
}

void VideoQuality::WriteResultsToBigVectors(FrameData& cFrame, 
											int numOfAttributesInputVector, 
											std::string& input_file,
											vector<vector<double>>& bigVectorOfResults, 
											vector<vector<double>>& bigVectorOfOtherResults,
											vector<string>& bigVectorOfFileNames)
{
	if(bigVectorOfResults.size() == 0)
	{
		for(int i =0 ; i< numOfAttributesInputVector; i++)
		{
			vector<double> result;
			bigVectorOfResults.push_back(result);
		}
	}

	if(cFrame.resizedImage_noise3_vector[0]>1.8)
			cFrame.resizedImage_noise3_vector[0] = 1.8;

	bigVectorOfResults[0].push_back(cFrame.resizedImage_noise3_vector[0]);
	bigVectorOfResults[1].push_back(cFrame.resizedImage_final_noise);
	bigVectorOfResults[2].push_back(cFrame.dynamicRange);
	bigVectorOfResults[3].push_back(cFrame.exposure*cFrame.contrast);
	bigVectorOfResults[4].push_back(cFrame.resizedImage_sharpness4);
	bigVectorOfResults[5].push_back(cFrame.resizedImage_details);
	bigVectorOfResults[6].push_back(cFrame.resizedImage_all_sharpness[0]);
	bigVectorOfResults[7].push_back(cFrame.resizedImage_all_sharpness[1]);
	bigVectorOfResults[8].push_back(cFrame.resizedImage_all_sharpness[2]);
	bigVectorOfResults[9].push_back(cFrame.resizedImage_all_sharpness[3]);
	bigVectorOfResults[10].push_back(cFrame.resizedImage_all_sharpness[4]);
	bigVectorOfResults[11].push_back(cFrame.resizedImage_all_sharpness[5]);
	bigVectorOfResults[12].push_back(cFrame.resizedImage_all_sharpness[6]);
	bigVectorOfResults[13].push_back(cFrame.resizedImage_all_sharpness[7]);
	bigVectorOfResults[14].push_back(cFrame.resizedImage_all_sharpness[8]);

	bigVectorOfFileNames.push_back(input_file);

	for(int i=0; i < 10; i++)
	{
		vector<double> result;
		bigVectorOfOtherResults.push_back(result);
	}

	bigVectorOfOtherResults[0].push_back(cFrame.resolution);
	bigVectorOfOtherResults[1].push_back(cFrame.resizedImage_noise2_vector[0]);
	bigVectorOfOtherResults[2].push_back(cFrame.resizedImage_noise2_vector[1]);
	bigVectorOfOtherResults[3].push_back(cFrame.colorWarmth);	
	bigVectorOfOtherResults[4].push_back(cFrame.saturation);
	bigVectorOfOtherResults[5].push_back(cFrame.blockiness);
	bigVectorOfOtherResults[6].push_back(cFrame.percentageOverExposure);
	bigVectorOfOtherResults[7].push_back(cFrame.percentageUnderExposure);
	bigVectorOfOtherResults[8].push_back(cFrame.resizedImage_noise2_vector[2]);
	bigVectorOfOtherResults[9].push_back(cFrame.resizedImage_noise3_vector[1]);

	//total of 25 
	for(int i = 0; i < cFrame.resizedImage_gammaMatrix_vector.size(); i++)
	{
		vector<double> result;
		result.push_back(cFrame.resizedImage_gammaMatrix_vector[i]);
		bigVectorOfOtherResults.push_back(result);
	}
}

void VideoQuality::WriteToCSV(  vector<double> vectorMOS, 
								vector<double> vectorBoundedMOS, 
								vector<double> modelVector,
								vector<vector<double>>& bigVectorOfResults, 
								vector<vector<double>>& bigVectorOfOtherResults, 
								vector<string>& bigVectorOfFileNames)
{
	string filename = "data.csv";
	string tempStr;

	for(int i = 0; i < bigVectorOfFileNames.size(); i++)
	{
		vector<string> record;

		record.push_back(bigVectorOfFileNames[i]);

		if( modelVector.size() > 0)
		{
			record.push_back("SentToModel->");
			for(int j = 0; j < modelVector.size(); j++)
			{
				tempStr = to_string(modelVector[j]);
				record.push_back(tempStr);
			}
		}

		record.push_back("SentToR->");
		for(int j = 0; j < bigVectorOfResults.size(); j++)
		{
			tempStr = to_string(bigVectorOfResults[j][i]);
			record.push_back(tempStr);
		}

		record.push_back("ForStudyOnly->");
		for(int j = 0; j < bigVectorOfOtherResults.size(); j++)
		{	
			tempStr = to_string(bigVectorOfOtherResults[j][i]);
			record.push_back(tempStr);
		}

		record.push_back("RawMOS->");
		if(vectorMOS.size() >= (i+1))
		{
			tempStr = to_string(vectorMOS[i]);
			record.push_back(tempStr);
		}

		record.push_back("BoundedMOS->");
		if(vectorBoundedMOS.size() >= (i+1))
		{
			tempStr = to_string(vectorBoundedMOS[i]);
			record.push_back(tempStr);
		}

		PrintStringVectorToFile(filename, record, true);
	}

}

void VideoQuality::Resize_image(Mat& gray_image,Mat& resized_GrayImage)
{
	cv::resize(gray_image,resized_GrayImage,Size(),0.5,0.5,INTER_CUBIC);
}



void VideoQuality::populate_rgamMap()
{
	for(double k=0.2 ; k<10.0;k=k+.001)
	{
		rgamMap[k]=(std::pow(Gamma(2/k),2)/((Gamma(1/k)*Gamma(3/k))));
	}
}

void VideoQuality::CalculateAllParamValues(Mat& subimage,double& alpha,double& betaL,double& betaR)
{
	double SumOfPixelVal=0, SumOfPixelValSquared=0;
	int totalCount=0;
	int LCnt=0,RCnt=0;
	double sum4LeftSTD=0.0,sum4RightSTD=0.0;
	map<double,double>::iterator rgamMapIter;
	for(int r1=0; r1<subimage.size().height;r1++)
	{
		for(int c1=0;c1<(subimage.size().width);c1++)
		{
			double PixelVal=subimage.at<double>(r1,c1);
			SumOfPixelVal+=abs(PixelVal);
			SumOfPixelValSquared+=pow(PixelVal,2);
			totalCount++;
			if(PixelVal<0)
			{
				sum4LeftSTD+=(PixelVal)*(PixelVal);
				LCnt++;
			}
			if(PixelVal>0)
			{
				sum4RightSTD+=(PixelVal)*(PixelVal);
				RCnt++;
			}
		}
	}
	double LInsertVal=sqrt(sum4LeftSTD/LCnt);
	double RInsertVal=sqrt(sum4RightSTD/RCnt);
	double gHatInsertVal=(LInsertVal/RInsertVal);
	double p=(std::pow((SumOfPixelVal/totalCount),2));
	double pp=(SumOfPixelValSquared/totalCount);
	double rHatInsertVal=(std::pow((SumOfPixelVal/totalCount),2))/(SumOfPixelValSquared/totalCount);
	double rNormInsertVal=(rHatInsertVal*(pow(gHatInsertVal,3)+1)*(gHatInsertVal+1))/(pow((pow(gHatInsertVal,2)+1),2));

	int debug = rgamMap.size();
	if(rgamMap.size() == 0) // if rgamMap is not already initialized
	{
		populate_rgamMap();
	}; 
	debug = rgamMap.size();

	double minVal=0,minPos=0,newMinVal=0;
	int firstCnt=0;
	for(rgamMapIter=rgamMap.begin();rgamMapIter!=rgamMap.end();rgamMapIter++)
	{ 
		newMinVal=pow(((rgamMapIter->second) -(rNormInsertVal)),2);
		if(firstCnt==0)
		{
			minVal=newMinVal;
			minPos=rgamMapIter->first;
			firstCnt++;
		}
		if(newMinVal<minVal)
		{
			minVal=newMinVal;
			minPos=rgamMapIter->first;
		}
	}

	alpha=minPos;
	betaL=LInsertVal*(sqrt((Gamma(1/alpha)/Gamma(3/alpha))));
	if(cvIsNaN(betaL))
	{betaL=0;}
	betaR=RInsertVal*(sqrt((Gamma(1/alpha)/Gamma(3/alpha))));
	if(cvIsNaN(betaR))
	{betaR=0;}

}


bool VideoQuality::PrintStringVectorToFile(string fileName, std::vector<string> myVector, bool putSeperators)
{
	ofstream tempOutfile1;
	tempOutfile1.open (fileName.c_str(),ios::app);
	if(!tempOutfile1.is_open())
	{
		cout<<"Can not open file. Please verify the path and filename!"<<endl;
		return FALSE;
	}

	stringstream tempSS1;
	string tempString;

	for(int i = 0; i < myVector.size(); i++)
	{
		tempString = myVector[i];
		if(putSeperators)	tempSS1 << tempString << ", ";
		else				tempSS1 << tempString << " ";
	}
	tempSS1<<endl;		

	tempOutfile1 << tempSS1.str();
	tempOutfile1.close();
}



const void VideoQuality::Calculate_Noise1(IplImage *img,double &Noise_val1,double &avg_activity)
{
	Mat input(img);
	Mat input_64(input.size().height,input.size().width,CV_64F);
	input.convertTo(input_64,CV_64F);		

	Mat channel[3];
	split(input_64, channel);
	int Gaussian_window=ceil(channel[0].rows/120.0);
	if(Gaussian_window%2 ==0)
	{
		Gaussian_window=Gaussian_window+1;
	}
	Mat sigmaRed,sigmaBlue,sigmaGreen;
	Calculate_Sigma(channel[0],Gaussian_window,sigmaBlue);
	Calculate_Sigma(channel[1],Gaussian_window,sigmaGreen);
	Calculate_Sigma(channel[2],Gaussian_window,sigmaRed);


	Mat AverageSigma=(sigmaRed+sigmaBlue+sigmaGreen)/3;
	Scalar  meanv=mean(AverageSigma);
	double noiseavg= meanv.val[0];
	vector <double> pixelVector;
	for(int i = 0; i < AverageSigma.cols; i++)
	{
		for(int j = 0; j < AverageSigma.rows; j ++)
		{
			double val= AverageSigma.at<double>(j,i);
			if(val>0.00001)
			{
				pixelVector.push_back(AverageSigma.at<double>(j,i));
			}
		}
	}

	std::sort(pixelVector.begin(),pixelVector.end());

	Mat pixelVectorToMat(pixelVector,true);

	meanv=mean(pixelVectorToMat);
	avg_activity= meanv.val[0];

	int  f=floor(0.005*pixelVector.size());
	vector <double> NewpixelVector;

	for(int i=0;i<f;i++)
	{
		NewpixelVector.push_back(pixelVector.at(i));
	}
	Mat NewpixelVectorToMat(NewpixelVector,true);
	Scalar  meanv2=mean(NewpixelVectorToMat);

	Noise_val1= meanv2.val[0];

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//HeatMap Methods

void VideoQuality::generateOutputfileNameHeatmap(string input,string attach, string &HeatMapDirecorty_Path, string& output)
{
	string outputname=input;
	int fileExtPosForDot = outputname.find_last_of(".");
	int fileExtPos = outputname.find_last_of("\\");

	String tmp_resFileName="";
	if (fileExtPosForDot >= 0 &&fileExtPos>=0 )
	{
		tmp_resFileName = outputname.substr(fileExtPos, fileExtPosForDot);
	}
	fileExtPosForDot = tmp_resFileName.find_last_of(".");
	if (fileExtPosForDot >= 0 )
	{
		tmp_resFileName = tmp_resFileName.substr(0, fileExtPosForDot);
	}
	output= HeatMapDirecorty_Path+ tmp_resFileName + attach;

}

string VideoQuality::CreateHeatmapDirectory(string video_file)
{
	string OutputDir=video_file;
	int fileExtPos = OutputDir.find_last_of("\\");
	if (fileExtPos >= 0)
	{
		String tmp_resFileName = OutputDir.substr(0, fileExtPos);
		OutputDir= tmp_resFileName+"\\HeatMap";
	}
	wstring wideusername;
	for(int i = 0; i < OutputDir.length(); ++i)
		wideusername += wchar_t( OutputDir[i] );
	const wchar_t* h1=(wideusername.c_str());
	bool retu = CreateDirectory(h1,NULL);
	if(retu==false)
	{
		if(GetLastError()==ERROR_ALREADY_EXISTS)
		{

		}
	}

	return OutputDir;
}

void VideoQuality::CreateHeatMap(Mat& imgMat,string Outputfile)
{
	double min,max;
	cv::minMaxIdx(imgMat, &min, &max);
	cv::Mat adjMap;
	cv::convertScaleAbs(imgMat, adjMap, 255 / max);

	Mat dst;
	applyColorMap(adjMap,dst,2);
	imwrite(Outputfile,dst);
	dst.release();
	adjMap.release();

}



/////Heat map methods called from Helper

void VideoQuality::Intensity_HeatMap(std::string& fileName, std::string& directory)
{ 
	IplImage* frame;
	string video_file=fileName;
	frame=cvLoadImage(video_file.c_str());
	string HeatMapPath= CreateHeatmapDirectory(video_file);

	Mat imgMat(frame);
	string outputFile="";
	generateOutputfileNameHeatmap(fileName,"_HeatMap_Intensity.jpg",directory,outputFile);
	CreateHeatMap(imgMat,outputFile);
	cvReleaseImage(&frame);
	frame=0;
	imgMat.release();

}

void VideoQuality::Contrast_HeatMap(std::string& video_file)
{
	IplImage* frame;
	frame=cvLoadImage(video_file.c_str());
	string HeatMapPath=CreateHeatmapDirectory(video_file);

	vector<vector<double>> totalvec;
	double con=Calculate_Contrast(frame,1,totalvec);
	// for heat map
	int col_count=totalvec[0].size();
	Mat imgMat(totalvec.size(),col_count,CV_64F);
	for(int i=0;i<totalvec.size();i++)
	{
		for(int j=0;j<totalvec[i].size();j++)
		{
			imgMat.at<double>(i,j)= (totalvec[i].at(j)); 
		}
	}
	string outputFile="";
	generateOutputfileNameHeatmap(video_file,"_HeatMap_Contrast.jpg",HeatMapPath,outputFile);
	CreateHeatMap(imgMat,outputFile);
	cvReleaseImage(&frame);
	frame=0;
	totalvec.clear();
	imgMat.release();

}

void VideoQuality::Sharpness_HeatMap(IplImage *frame)//std::string& fileName)
{
	Mat image(frame);
	cv::Mat sharpness;
	cv::Mat gray_image;

	cvtColor( image, gray_image, CV_RGB2GRAY );
	cv::Mat sobelH_OnImg;
	cv::Mat sobelV_OnImg;
	cv::Sobel(gray_image, sobelH_OnImg, -1, 0, 1, 3, 1, 0, BORDER_CONSTANT);
	cv::Sobel(gray_image, sobelV_OnImg, -1, 1, 0, 3, 1, 0, BORDER_CONSTANT);
	cv::Mat abs_sobelH_OnImg = cv::abs(sobelH_OnImg*(-1));  
	cv::Mat abs_sobelV_OnImg = cv::abs(sobelV_OnImg*(-1));  	
	sharpness = (abs_sobelH_OnImg + abs_sobelV_OnImg)*0.5;

	cv::Mat rgb_image;
	cvtColor( sharpness, rgb_image, CV_GRAY2RGB ); // 24bit image needed (instead of 8bit) for pdf report !
	CreateHeatMap(rgb_image,outputSharpness);

	image.release();
	sharpness.release();
	gray_image.release();
	rgb_image.release();

}

bool VideoQuality::DrawHistogram(IplImage *frame)//std::string& fileName)
{
	Mat src(frame);

	if( !src.data )
	{ return -1; }

	/// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes;
	split( src, bgr_planes );

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };

	bool uniform = true; 
	bool accumulate = false;

	Mat b_hist, g_hist, r_hist, gray_hist;

	cv::Mat gray_image;
	cvtColor( src, gray_image, CV_RGB2GRAY );

	/// Compute the histograms:
	calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist,    1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist,    1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist,    1, &histSize, &histRange, uniform, accumulate );
	calcHist( &gray_image   , 1, 0, Mat(), gray_hist, 1, &histSize, &histRange, uniform, accumulate );

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histSize );

	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist,    b_hist,    0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(g_hist,    g_hist,    0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(r_hist,    r_hist,    0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(gray_hist, gray_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() ); 

	/// Draw for each channel
	for( int i = 1; i < histSize; i++ )
	{

		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
			Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
			Scalar( 255, 0, 0), 2, 8, 0  );
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
			Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
			Scalar( 0, 255, 0), 2, 8, 0  );
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
			Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
			Scalar( 0, 0, 255), 2, 8, 0  );
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(gray_hist.at<float>(i-1)) ) ,
			Point( bin_w*(i), hist_h - cvRound(gray_hist.at<float>(i)) ),
			Scalar( 127, 127, 127), 2, 8, 0  );

	}

	imwrite(outputDR,histImage);

	return 0;
}

void getScaleToMonitor(int r_orig, int c_orig, int& r_final, int& c_final) 
{
	const int monitor_rows = 1200;
	const int monitor_cols = 1920;

	int c_temp = monitor_rows * c_orig / r_orig;
	if( c_temp <= monitor_cols)
	{
		r_final = monitor_rows;
		c_final = c_temp;
	}
	else
	{
		r_final = monitor_cols * r_orig / c_orig;
		c_final = monitor_cols;
	}

}

FrameData* VideoQuality::Analyze_Frame(void* imgVoidPt)
{
	std::string sourceFileName = "", category="Undefined", mosModel="GenericMosLibSVM", directory = "";
	return AnalyzePhoto(sourceFileName, category, mosModel, directory, true, imgVoidPt);
}

vector<double> VideoQuality::ValueSummary(vector<double> values)
{
	Mat valueMat = Vector2Mat(values, values.size(), 1);

	Scalar std, mean;
	cv::meanStdDev(valueMat, mean, std);

	double std_err = std.val[0] / pow(values.size(), 0.5);

	vector<double> _return;
	_return.push_back(mean.val[0]);
	_return.push_back(std_err);	

	return _return;
}



FrameData* VideoQuality::AnalyzePhoto(std::string& sourceFileName, std::string& category, std::string& mosModel, std::string& directory, bool onCloud, void* imgVoidPt)
{
	vector<vector<double>> bigVectorOfResults;
	vector<vector<double>> bigVectorOfOtherResults;
	vector<string> bigVectorOfFileNames;

	int numOfAttributesInRVector = 15;
	float  start_time;

	//Check to see if imgVoidPt is set to 0. 
	//If so, then sourceFileName and directory are non-empty and use them.
	//Otherwise, this is to process a frame in imgVoidPt coming from a video

	IplImage* frame = nullptr;
	ifstream ifile (sourceFileName);

	if(imgVoidPt != 0)
	{
		frame = (IplImage*) imgVoidPt;
	}
	else
	{
		frame=cvLoadImage(sourceFileName.c_str());
	}

	FrameData *frameData = new FrameData();		
	map<int,FrameData> frame_data_map;
	float time_frame_val=0;
	if (frame !=nullptr)								//if the frame is valid.
	{
		setResolution(frame->width,frame->height);

		if(frame)
		{
			//resize original to a smaller resolution;
			int r_resized, c_resized;
			getScaleToMonitor(frame->height, frame->width, r_resized, c_resized);
			IplImage *frame_resized = cvCreateImage( cvSize(c_resized, r_resized), frame->depth, frame->nChannels);
			cvResize(frame, frame_resized);

			double contrast_val=0, saturation_val=0, exposure_val=0, noise_val=0, sharpness_val0=0, blockiness_val=0, dynamicRange_val=0, hue_val=0;
			double cr_val=0, cb_val=0, sharpness_val1=0, sharpness_val2=0, sharpness_val3=0, resizedImage_sharpness_val3=0, sharpness_val4=0, resizedImage_sharpness_val4=0, resizedImage_sharpness_detail=0;
			double noise_val1=0, resizedImage_noise_val1=0, avg_activity=0, resizedImage_avg_activity=0, colorWarmth=0;
			vector<double> sharpness_val5, resizedImage_sharpness_val5, noise2_vector, resizedImage_noise2_vector;
			double percentageOverExposure = 0, percentageUnderExposure = 0;			
			vector<vector<double>> totalvec; 

			double start_time1  = clock();
						
			vector<double> resizedImage_noise3_vector = VideoQuality::Calculate_Noise3(frame_resized);
			vector<double> resizedImage_gammaMatrix_vector = VideoQuality::Compute_GammaMatrix(frame_resized, 36);
						
			if(!onCloud)
			{
				task_group FeatureTask;		

				/*1*/ FeatureTask.run([frame_resized,&contrast_val]{ 
							VideoQuality::Calculate_Contrast(frame_resized,contrast_val);});
				/*2*/ FeatureTask.run([frame_resized,&saturation_val]{ 
							VideoQuality::Calculate_Saturation(frame_resized,saturation_val);});
				/*3*/ FeatureTask.run([frame_resized,&exposure_val]{ 
							VideoQuality::Calculate_Exposure(frame_resized,exposure_val);});
				/*4*/ FeatureTask.run([frame_resized,&dynamicRange_val]{ 
							VideoQuality::Calculate_DynamicRange(frame_resized,dynamicRange_val);});
				/*5*/ FeatureTask.run([frame_resized,&resizedImage_sharpness_val3]{ 
							VideoQuality::Calculate_Sharpness3(frame_resized,resizedImage_sharpness_val3);});
				/*6*/ FeatureTask.run([frame_resized,&resizedImage_noise_val1,&resizedImage_avg_activity]{ 
							VideoQuality::Calculate_Noise1(frame_resized,resizedImage_noise_val1,resizedImage_avg_activity);});
				/*7*/ FeatureTask.run([frame_resized,&resizedImage_sharpness_val4, &resizedImage_sharpness_detail]{ 
							VideoQuality::Calculate_Sharpness4(frame_resized,resizedImage_sharpness_val4, resizedImage_sharpness_detail, false);});	 // NEW-ALGO !!				 
				/*8*/ FeatureTask.run([frame_resized,&colorWarmth]{
							VideoQuality::Calculate_ColorWarmth(frame_resized,colorWarmth);	});
				/*9*/ FeatureTask.run([frame_resized,&percentageOverExposure, &percentageUnderExposure]{
							Calculate_OverAndUnderExposure(frame_resized, percentageOverExposure, percentageUnderExposure, false);});
				/*10*/ FeatureTask.run([frame_resized,&resizedImage_sharpness_val5]{ 
							VideoQuality::Calculate_Sharpness5(frame_resized, resizedImage_sharpness_val5); });// returns 9 values															
				/*11*/ FeatureTask.run([frame_resized,&resizedImage_noise2_vector, directory, mosModel]{ 
							VideoQuality::Calculate_Noise2(frame_resized, resizedImage_noise2_vector, directory, mosModel); }); // returns 3 values
						
				

				FeatureTask.wait();
			}
			else
			{
				
			// Current Set !
			/*1*/ contrast_val					= Calculate_Contrast(frame_resized,0,totalvec);		
			/*2*/ saturation_val				= Calculate_Saturation(frame_resized);
			/*3*/ exposure_val					= Calculate_Exposure(frame_resized);
			/*4*/ dynamicRange_val				= Calculate_DynamicRange(frame_resized);
			/*5*/ resizedImage_sharpness_val3	= Calculate_Sharpness3(frame_resized);
			/*6*/ resizedImage_avg_activity		= Calculate_Noise1(frame_resized, resizedImage_noise_val1);
			/*7*/ resizedImage_sharpness_val4	= Calculate_Sharpness4(frame_resized, resizedImage_sharpness_detail, false);
			/*8*/ colorWarmth					= Calculate_ColorWarmth(frame_resized);	
			/*9*/ Calculate_OverAndUnderExposure(frame_resized, percentageOverExposure, percentageUnderExposure);

				/*10*/ resizedImage_sharpness_val5		= Calculate_Sharpness5(frame_resized);							// returns 9 values
				/*11*/ resizedImage_noise2_vector		= Calculate_Noise2    (frame_resized, directory, mosModel);		// returns 3 values

				
			}

			double start_time2  = clock();			
			double timeLapse  = (start_time2 - start_time1)/CLOCKS_PER_SEC;

						
			
			if(noise2_vector.size() == 3)
			{
				bool MISSINGVALUEFOUND = ((noise2_vector[0] == -3.142) ? true : false);			

				if(MISSINGVALUEFOUND)
				{
					if(category.compare(Constants::LANDSCAPE) == 0)
					{
						noise2_vector[0] = 1.03648997222222;
						noise2_vector[1] = 0.386365500000000;
						noise2_vector[2] = 0.0900289444444445;
					}
					else if (category.compare(Constants::DINNER_PLATE) == 0)
					{
						noise2_vector[0] = 1.21754980000000;
						noise2_vector[1] = 0.426604819047619;
						noise2_vector[2] = 0.0981914095238095;
					}
					else if (category.compare(Constants::WALL_HANGING) == 0)
					{
						noise2_vector[0] = 1.20458363846154;
						noise2_vector[1] = 0.426268538461538;
						noise2_vector[2] = 0.0990906307692307;
					}
					else if (category.compare(Constants::NIGHT_SHOT) == 0)
					{
						noise2_vector[0] = 0.936863294573643;
						noise2_vector[1] = 0.367852596899225;
						noise2_vector[2] = 0.160716085271318;
					}
					else
					{	
						noise2_vector[0] = 1.09583661864407;
						noise2_vector[1] = 0.401247616525424;
						noise2_vector[2] = 0.113659705508475;
					}
				}
			}

			if(resizedImage_noise2_vector.size() == 3)
			{
				bool MISSINGVALUEFOUND_RESIZED = ((resizedImage_noise2_vector[0] == -3.142) ? true : false);			

				if(MISSINGVALUEFOUND_RESIZED)
				{
					if(category.compare(Constants::LANDSCAPE) == 0)
					{
						resizedImage_noise2_vector[0] = 1.03648997222222;
						resizedImage_noise2_vector[1] = 0.386365500000000;
						resizedImage_noise2_vector[2] = 0.0900289444444445;
					}
					else if (category.compare(Constants::DINNER_PLATE) == 0)
					{
						resizedImage_noise2_vector[0] = 1.21754980000000;
						resizedImage_noise2_vector[1] = 0.426604819047619;
						resizedImage_noise2_vector[2] = 0.0981914095238095;
					}
					else if (category.compare(Constants::WALL_HANGING) == 0)
					{
						resizedImage_noise2_vector[0] = 1.20458363846154;
						resizedImage_noise2_vector[1] = 0.426268538461538;
						resizedImage_noise2_vector[2] = 0.0990906307692307;
					}
					else if (category.compare(Constants::NIGHT_SHOT) == 0)
					{
						resizedImage_noise2_vector[0] = 0.936863294573643;
						resizedImage_noise2_vector[1] = 0.367852596899225;
						resizedImage_noise2_vector[2] = 0.160716085271318;
					}
					else
					{	
						resizedImage_noise2_vector[0] = 1.09583661864407;
						resizedImage_noise2_vector[1] = 0.401247616525424;
						resizedImage_noise2_vector[2] = 0.113659705508475;
					}
					resizedImage_noise2_vector[2] = std::numeric_limits<double>::quiet_NaN();
				}
			}
			//---------------FILL MISSING values END !!						

			// To create the heatMap Folder. If it exists then nothing is done			
			if(directory.compare("dummy")!=0)
			{
				int uniqueNo = rand();
				
				generateOutputfileNameHeatmap(sourceFileName, std::to_string(uniqueNo) + "_Sharpness_HeatMap.jpg", directory, outputSharpness);			
				generateOutputfileNameHeatmap(sourceFileName, std::to_string(uniqueNo) + "_DynamicRangeHistogramFile.jpg", directory, outputDR);
			}

			double final_noise=0;
			if(noise_val1 ==0 ||avg_activity==0 ||sharpness_val3==0)
			{
				final_noise=0;			
			}
			else
			{
				final_noise=100000*((noise_val1/avg_activity)/(sharpness_val3*VideoQuality::resolution));
			}

			double resizedImage_final_noise=0;
			if(resizedImage_noise_val1 ==0 || resizedImage_avg_activity==0 || resizedImage_sharpness_val3==0)
			{
				resizedImage_final_noise=0;			
			}
			else
			{
				resizedImage_final_noise=((resizedImage_noise_val1/resizedImage_avg_activity)/(resizedImage_sharpness_val3));
			}

			//Add last 5 values to calculate new detail measure.
			double tempSum = 0.0;
			int tempSize = resizedImage_gammaMatrix_vector.size() - 5;
			for(int i = tempSize; i < resizedImage_gammaMatrix_vector.size(); i++)
			{

				tempSum += resizedImage_gammaMatrix_vector[i];
			}
			//calculate the average of this 5 values and save as multiScaleTextureAcutance
			double tempMultiScaleTextureAcutance = tempSum/5.0;

			//insert the values in the FrameData
			frameData->setValues(	outputSharpness,
									outputDR,
									VideoQuality::resolution,
									0,
									time_frame_val,
									contrast_val,
									saturation_val,
									exposure_val,
									noise_val,
									sharpness_val0,
									blockiness_val,
									dynamicRange_val,
									colorWarmth,
									percentageOverExposure,
									percentageUnderExposure,
									hue_val,
									cr_val,
									cb_val,
									sharpness_val1,
									sharpness_val2,
									sharpness_val3,
									resizedImage_sharpness_val3,
									resizedImage_sharpness_detail,
									noise_val1,
									avg_activity,
									sharpness_val4,
									resizedImage_sharpness_val4,
									sharpness_val5,
									resizedImage_sharpness_val5,
									noise2_vector,
									resizedImage_noise2_vector,
									resizedImage_noise3_vector,
									resizedImage_gammaMatrix_vector,
									final_noise,
									resizedImage_final_noise,
									tempMultiScaleTextureAcutance);
			
			WriteResultsToBigVectors(	*frameData, 
										numOfAttributesInRVector, 
										sourceFileName, 
										bigVectorOfResults, 
										bigVectorOfOtherResults, 
										bigVectorOfFileNames);					

			double mos = MOS(category, mosModel, bigVectorOfResults, bigVectorOfOtherResults, bigVectorOfFileNames, directory, onCloud);

			
			bigVectorOfResults.clear();
			bigVectorOfOtherResults.clear();
			bigVectorOfFileNames.clear();


			frameData->mos = mos;
			frame_data_map[0]=*frameData;	


			if(sourceFileName != "")
			{
				Sharpness_HeatMap(frame_resized);
				DrawHistogram(frame_resized);
			}

		}

		time_frame_val+=0;
		frame=0;

		frame_data_map.clear();
		frame_data_map.clear();
	}
	else
	{
		cout<<"Input  file does not exist. Please verify."<<endl;
	}
	
	return frameData;
}

double VideoQuality::MOS(std::string& category, 
						 std::string& mosModel, 
						 vector<vector<double>>& bigVectorOfResults, 
						 vector<vector<double>>& bigVectorOfOtherResults, 
						 vector<string>& bigVectorOfFileNames, 
						 std::string directory, bool onCloud)
{
	vector<double> predictions;
	
	int uniqueNo = rand();

	string inputFile_string = directory + "\\" + std::to_string(uniqueNo) + "photoFeatures.txt";
	const char* inputFile = inputFile_string.c_str();
	string outputFile_string = directory + "\\" + std::to_string(uniqueNo) + "mos.txt";
	const char* outputFile = outputFile_string.c_str();

	remove(inputFile);  // delete entries of last run !
	remove(outputFile); // delete predictions of last run !

	int noOfAttributes = bigVectorOfResults.size();
	int noOfEntries = bigVectorOfResults[0].size();

	//Check for missing values and replace with defaults
	bool MISSINGVALUEFOUND = bigVectorOfResults[1][0] == -3.142 ? true : false;

	string scriptFile, train_file, model_file;
	
	if(mosModel.compare(Constants::CategoryMOS_LibSvm)== 0)
	{
		if((category.compare(Constants::LANDSCAPE) == 0) || (category.compare("Outdoor Day - Landscape")==0))
		{
			train_file = "MOSlandscapeTrainingData.txt";
			model_file = "MOSModelData_forLandscape.txt";	
		}
		else if ((category.compare(Constants::DINNER_PLATE) == 0) || (category.compare("Indoor - Arrangements")==0))
		{			
			train_file = "MOSfoodTrainingData.txt";
			model_file = "MOSModelData_forFood.txt";
		}
		else if ((category.compare(Constants::WALL_HANGING) == 0)  || (category.compare("Indoor - Wall Hanging ")==0))
		{
			train_file = "MOSwallTrainingData.txt";
			model_file = "MOSModelData_forWall.txt";
		}
		else if ((category.compare(Constants::NIGHT_SHOT) == 0)   || (category.compare("Outdoor Night - Landmark")==0))
		{
			train_file = "MOSnightTrainingData.txt";
			model_file = "MOSModelData_forNight.txt";
		}
		else
		{	
			train_file = "MOSallTrainingData.txt";
			model_file = "MOSModelData_forAll.txt";
		}
	}
	else if (mosModel.compare(Constants::GenericMOS_LibSvm)== 0)
	{
		train_file = "MOSallTrainingData.txt";
		model_file = "MOSModelData_forAll.txt";
	}	

	vector<double> modelVector;
	for(int j = 0; j < noOfAttributes; j++)
	{
		modelVector.push_back(std::log(1 + bigVectorOfResults[j][0]));		
	}

	//bigVectorOfOtherResults indexes 3-4, 6-7, 9-end
	modelVector.push_back(std::log(1+bigVectorOfOtherResults[3][0]));
	modelVector.push_back(std::log(1+bigVectorOfOtherResults[4][0]));

	modelVector.push_back(std::log(1+bigVectorOfOtherResults[6][0]));
	modelVector.push_back(std::log(1+bigVectorOfOtherResults[7][0]));

	for(int i=9;i<bigVectorOfOtherResults.size();i++)
		modelVector.push_back(std::log(1+bigVectorOfOtherResults[i][0]));

	for(int i=6;i<15;i++)
		modelVector.push_back(std::log(1 + bigVectorOfResults[i][0]/exp(bigVectorOfResults[0][0])));		

	//8 different new featuers R1-R8 calculated here and then pushed to modelVector
	double R[8];

	double tempSum = (bigVectorOfOtherResults[29][0] + bigVectorOfOtherResults[30][0] + bigVectorOfOtherResults[31][0] + bigVectorOfOtherResults[32][0])/4;
	R[0] = bigVectorOfOtherResults[28][0]/(1+0.5*(tempSum+bigVectorOfOtherResults[33][0]));
	R[1] = tempSum/(1+bigVectorOfOtherResults[33][0]);

	tempSum = (bigVectorOfOtherResults[23][0] + bigVectorOfOtherResults[24][0] + bigVectorOfOtherResults[25][0] + bigVectorOfOtherResults[26][0])/4;
	R[2] = bigVectorOfOtherResults[22][0]/(1+0.5*(tempSum+bigVectorOfOtherResults[27][0]));
	R[3] = tempSum/(1+bigVectorOfOtherResults[27][0]);

	tempSum = (bigVectorOfOtherResults[17][0] + bigVectorOfOtherResults[18][0] + bigVectorOfOtherResults[19][0] + bigVectorOfOtherResults[20][0])/4;
	R[4] = bigVectorOfOtherResults[16][0]/(1+0.5*(tempSum+bigVectorOfOtherResults[21][0]));
	R[5] = tempSum/(1+bigVectorOfOtherResults[21][0]);

	tempSum = (bigVectorOfOtherResults[11][0] + bigVectorOfOtherResults[12][0] + bigVectorOfOtherResults[13][0] + bigVectorOfOtherResults[14][0])/4;
	R[6] = bigVectorOfOtherResults[10][0]/(1+0.5*(tempSum+bigVectorOfOtherResults[14][0]));
	R[7] = tempSum/(1+bigVectorOfOtherResults[15][0]);


	for(int i=0;i<8;i++)
		modelVector.push_back(log(1+R[i]));


	double mos = 0.0;

	if(mosModel.compare(Constants::GenericMOS_LibSvm)== 0 || mosModel.compare(Constants::CategoryMOS_LibSvm)== 0) 
	
	{																																																				
 
		// --------------------SVM Test using NO Files----------------------------
		vector<svm_node *> testDataSVM;
		int max_nr_attr = modelVector.size(); // length of attribute string	
		struct svm_node *x = (struct svm_node *) malloc((max_nr_attr+1)*sizeof(struct svm_node));
		
		int j=0;
		for(; j < modelVector.size(); j++)
		{
			x[j].index = (j+1);
			x[j].value = modelVector[j];
		}
		x[j].index = -1; // termination indicator for libSVM
		testDataSVM.push_back(x);	

		//debugging
		FILE *fp = fopen("c:\\temp\\path.txt","wt");
		const char *name = model_file.c_str();
		if(fp==NULL) return NULL;
		fprintf(fp, "model file name = %s\n",model_file.c_str());
		fclose(fp);

		struct svm_model* input_model;
		if((input_model=svm_load_model(model_file.c_str()))==0)
		{	
			fprintf(stderr,"can't open model file %s\n", model_file.c_str());	
			return(-3.142); 		
		}
		
		SVMClass svmTestObject;
		std::vector<double> predictedLabels;
		std::vector<std::vector<double>> classProbability;
		svmTestObject.main_svmPredictFileless(0, testDataSVM, input_model, predictedLabels, classProbability);

		mos = predictedLabels[0];		
	}
	
	//Bound the MOS
	double boundMOS = mos;
	if(mos > 4.5) boundMOS = 4.5; // Hard Upper Bound on Individual Photos to 4.5
	if(mos < 1.0) boundMOS = 1.0; // Hard Lower Bound on Individual Photos to 1.

	vector<double> individualMOS;
	individualMOS.push_back(mos);

	vector<double> boundedMOS;
	boundedMOS.push_back(boundMOS);

	if(!onCloud)
	{
		WriteToCSV(individualMOS, boundedMOS, modelVector, bigVectorOfResults, bigVectorOfOtherResults, bigVectorOfFileNames);
	}

	return boundMOS;
}  