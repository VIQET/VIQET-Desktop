/*
Copyright © 2015 Intel Corporation
This program and the accompanying materials are made available under the terms of the Eclipse Public License v1.0, 
 which accompanies this distribution, and is available at http://www.eclipse.org/legal/epl-v10.html . https://github.com/viqet
* Contributors:
*    Intel Corporation - initial API and implementation and/or initial documentation
*/
#include "StdAfx.h"
#include "VideoQuality.h"
#include "UMHelper.h"


UMHelper::UMHelper(void)
{
	ptrVQ = VideoQuality::pGetInstance();
}

FrameData* UMHelper::AnalyzePhoto(std::string& sourceFileName, std::string& category, std::string& mosModel, std::string& directory, bool onCloud)
{
		return ptrVQ->AnalyzePhoto(sourceFileName, category, mosModel, directory, onCloud);
}

std::vector<double> UMHelper::ValueSummary(std::vector<double> values)
{
	return ptrVQ->ValueSummary(values);
}


FrameData* UMHelper::getFrameData()
{
	return ptrVQ->getFrameData();
}


double UMHelper::Calculate_Saturation(void* imageVoidPt)
{
	return ptrVQ->Calculate_Saturation(imageVoidPt);
}

void UMHelper::Calculate_Contrast(void *gimgVoidPt, double &Contrast_val)
{
	ptrVQ->Calculate_Contrast(gimgVoidPt, Contrast_val);
}


FrameData* UMHelper::Analyze_Frame(void* imgVoidPt)
{
	return ptrVQ->Analyze_Frame(imgVoidPt);
}