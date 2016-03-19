/*
Copyright © 2015 Intel Corporation
This program and the accompanying materials are made available under the terms of the Eclipse Public License v1.0, 
 which accompanies this distribution, and is available at http://www.eclipse.org/legal/epl-v10.html . https://github.com/viqet
* Contributors:
*    Intel Corporation - initial API and implementation and/or initial documentation
*/
#include <iostream>
#include <vector>

//using namespace cv;
using namespace std;

#pragma once

class FrameData{
	public:
		int frame_number;
		double contrast;
		double saturation;
		double exposure;
		double noise;
		double sharpness;
		double blockiness;
		double dynamicRange;
		double colorWarmth;
		double percentageOverExposure;
		double percentageUnderExposure;
		double time_frame;
		double hue;
		double cr;
		double cb;
		double sharpness1;
		double sharpness2;
		double sharpness3;
		double resizedImage_sharpness3;
		double sharpness4;
		double resizedImage_sharpness4;
		double resizedImage_details;
		double noise1;
		double avg_activity;
		
	    string SharpnessFile;
		string DRHistogramFile;
		double resolution;
		vector<double> all_sharpness;
		vector<double> resizedImage_all_sharpness;
		vector<double> noise2_vector;
		vector<double> resizedImage_noise2_vector;
		vector<double> resizedImage_noise3_vector;
		vector<double> resizedImage_gammaMatrix_vector;
		double final_sharpness_displayed;
		double final_noise;
		double resizedImage_final_noise;
		double mos;
		double multiScaleTextureAcutance;
		vector<double> NSS_features;
	//used by images
		void setValues(	string outputSharpness_File,
						string outputDR_File,
						double resolution_val,
						int frame_num,
						double time_frame_val,
						double contrast_val,
						double sat_val,
						double exposure_val,
						double noise_val,
						double sharpness_val,
						double blockiness_val,
						double dynamicRange_val,
						double colorWarmth_val,
						double percentageOverExposure_val,
						double percentageUnderExposure_val,
						double hue_val,
						double cr_val,
						double cb_val, 
						double sharpness1_val,
						double sharpness2_val,
						double sharpness3_val,
						double resizedImage_sharpness3_val,
						double resizedImage_details_val,
						double noise_val1,
						double avg_activity_val,
						double sharpness4_val,
						double resizedImage_sharpness4_val,
						vector<double> all_sharpness_val, 
						vector<double> resizedImage_all_sharpness_val,
						vector<double> noise2_vector_val,
						vector<double> resizedImage_noise2_vector_val,
						vector<double> resizedImage_noise3_vector_val,
						vector<double> resizedImage_gammaMatrix_vector_val,
						double final_noise_val,
						double resizedImage_final_noise_val,
						double msta,
						vector<double> NSS_features_val)
		{
		    SharpnessFile=outputSharpness_File;
			DRHistogramFile=outputDR_File;
			resolution=resolution_val;
			frame_number=frame_num;
			contrast	= contrast_val;
			saturation	= sat_val;
			exposure	= exposure_val;
			noise		= noise_val;
			sharpness	= sharpness_val;
			blockiness	= blockiness_val;
			dynamicRange= dynamicRange_val;
			colorWarmth= colorWarmth_val;
			percentageOverExposure = percentageOverExposure_val;
			percentageUnderExposure = percentageUnderExposure_val;
			time_frame=time_frame_val;
			hue=hue_val;
			cr=cr_val;
			cb=cb_val;
			sharpness1 = sharpness1_val;
			sharpness2 = sharpness2_val;
			sharpness3 = sharpness3_val;
			resizedImage_sharpness3 = resizedImage_sharpness3_val;
			resizedImage_details    = resizedImage_details_val;
			noise1	   = noise_val1;
			avg_activity =avg_activity_val;
			sharpness4 =sharpness4_val;
			resizedImage_sharpness4 = resizedImage_sharpness4_val;
			all_sharpness=all_sharpness_val;
			resizedImage_all_sharpness=resizedImage_all_sharpness_val;
			noise2_vector=noise2_vector_val;
			resizedImage_noise2_vector=resizedImage_noise2_vector_val;
			resizedImage_noise3_vector=resizedImage_noise3_vector_val;
			resizedImage_gammaMatrix_vector=resizedImage_gammaMatrix_vector_val;
			multiScaleTextureAcutance = msta;
			NSS_features=NSS_features_val;
			if(resizedImage_all_sharpness_val.size() == 9)
			{
				final_sharpness_displayed= (std::pow((
				resizedImage_all_sharpness_val.at(0)*
				resizedImage_all_sharpness_val.at(1)*
				resizedImage_all_sharpness_val.at(2)*
				resizedImage_all_sharpness_val.at(3)*
				resizedImage_all_sharpness_val.at(4)*
				resizedImage_all_sharpness_val.at(5)*
				resizedImage_all_sharpness_val.at(6)*
				resizedImage_all_sharpness_val.at(7)*
				resizedImage_all_sharpness_val.at(8)),1.0/9.0))/10000;
			} 
			else final_sharpness_displayed = 0;

			final_noise= final_noise_val;
			resizedImage_final_noise= resizedImage_final_noise_val;
		} 

		//used by video
		void setValues(double resolution_val,int frame_num,double time_frame_val,double contrast_val,double sat_val,double exposure_val,double noise_val,double sharpness_val,double blockiness_val,double dynamicRange_val,double hue_val,double cr_val,double cb_val, double sharpness1_val,double sharpness2_val,double sharpness3_val,double noise_val1,double avg_activity_val, double multiScaleTextureAcutance)
		{
			resolution=resolution_val;
			frame_number=frame_num;
			contrast	= contrast_val;
			saturation	= sat_val;
			exposure	= exposure_val;
			noise		= noise_val;
			sharpness	= sharpness_val;
			blockiness	= blockiness_val;
			dynamicRange= dynamicRange_val;
			time_frame=time_frame_val;
			hue=hue_val;
			cr=cr_val;
			cb=cb_val;
			sharpness1 =sharpness1_val;
			sharpness2 =sharpness2_val;
			sharpness3 =sharpness3_val;
			noise1		= noise_val1;
			avg_activity =avg_activity_val;			
		}
};

		