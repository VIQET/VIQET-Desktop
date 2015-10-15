/*
Copyright © 2015 Intel Corporation
This program and the accompanying materials are made available under the terms of the Eclipse Public License v1.0, 
 which accompanies this distribution, and is available at http://www.eclipse.org/legal/epl-v10.html . https://github.com/viqet
* Contributors:
*    Intel Corporation - initial API and implementation and/or initial documentation
*/
#ifdef UMHELPER_EXPORTS
#define UMHELPER_E __declspec(dllexport)
#else
#define  UMHELPER_E __declspec(dllimport)
#endif

#pragma once
#include <string>
#include <map>


class VideoQuality;
class FrameData;


class UMHELPER_E UMHelper
{
public:
	UMHelper(void);
	FrameData* AnalyzePhoto(std::string& sourceFileName, std::string& category, std::string& mosModel, std::string& directory, bool onCloud);
	std::vector<double> UMHelper::ValueSummary(std::vector<double> values);
	FrameData* getFrameData();
	
	double Calculate_Saturation(void *imageVoidPt);
	void Calculate_Contrast(void *gimgVoidPt, double &Contrast_val);
	FrameData* Analyze_Frame(void* imgVoidPt);
	
private:
	VideoQuality* ptrVQ;
};

