/*
Copyright © 2015 Intel Corporation
This program and the accompanying materials are made available under the terms of the Eclipse Public License v1.0, 
 which accompanies this distribution, and is available at http://www.eclipse.org/legal/epl-v10.html . https://github.com/viqet
* Contributors:
*    Intel Corporation - initial API and implementation and/or initial documentation
*/

// This is the main DLL file.

#include "VQHelper.h"
#include <iostream>
#include <fstream>
#include <vcclr.h>

using namespace Runtime::InteropServices;
using namespace VQHelper;



void MarshalTheVector(vector<double>& input,array<double>^% output)
{
	// get unmanaged values
	vector<double> values;
	for(int i=0;i<input.size();i++)
	{ 
		double val=input.at(i);
		values.push_back(val);
	}		
	output= gcnew array<double>(values.size());
	// cast to managed object type IntPtr representing an object pointer.
	System::IntPtr ptr = (System::IntPtr)&values[0];

	// copy data to managed array using System::Runtime::Interopservices namespace
	Marshal::Copy(ptr,output , 0, values.size());
}

void HelperClass::setDirectoryPathForFiles(String^ r)
{
	directoryPathForFiles=r;
}

HelperClass::HelperClass(String^ directoryName)
{
	std::string fileName(""), fileLocation(ManagedToUnmanagedString(directoryName));
	umHelper = new UMHelper();	
	this->directoryPathForFiles = directoryName + "\\";
	
}

HelperClass::~HelperClass()
{
	delete umHelper;
}

string HelperClass::ManagedToUnmanagedString(String^ s)
{
	const char* chars =(const char*)(Marshal::StringToHGlobalAnsi(s)).ToPointer();
	std::string os(chars);
	Marshal::FreeHGlobal(IntPtr((void*)chars));
	return os;
}

Parameter^ CreateParameter(String^ parameterName, double value, String^ displayPreference)
{
	Parameter^ parameter = gcnew Parameter();
	parameter->ParameterName = parameterName;
	parameter->Value = value;
	parameter->DisplayPreference = displayPreference;
	return parameter;
}

VisualizationImage^ CreateVisualizationImage(String^ visualizationName, String^ filePath)
{
	VisualizationImage^ visualizationImage = gcnew VisualizationImage();
	visualizationImage->Visualization = visualizationName;
	visualizationImage->FilePath = filePath;
	return visualizationImage;
}

int HelperClass::GetBackendVersion()
{
    // Backend version
	return 117;
}

Summary^ HelperClass::ValueSummary(List<double>^ values)
{
	std::vector<double> v;
	for(int i=0;i<values->Count;i++)
		v.push_back(values[i]);
	
	vector<double> calculatedSummary = umHelper->ValueSummary(v);	

	Summary^ summaryResult = gcnew Summary();
	summaryResult->mean = calculatedSummary[0];
	summaryResult->range = calculatedSummary[1];

	return summaryResult;
}

List<Summary^>^ HelperClass::ValueSummaryList(List<List<double>^>^ listOfvalues)
{
	List<Summary^>^ _return = gcnew List<Summary^>();

	for(int i =0; i < listOfvalues->Count; i++)
	{
		Summary^ summaryResult = ValueSummary(listOfvalues[i]);
		_return->Add(summaryResult);
	}
	return _return;
}

/*ResultSummaryComputed^ HelperClass::ResultSummary(ResultSummaryToCompute^ input)
{
	ResultSummaryComputed^ _return = gcnew ResultSummaryComputed();	
	
	_return->device = ValueSummary(input->device->values);
	_return->device->valueName = input->device->valueName;

	_return->categoryList = gcnew List<CategorySummary^>();

	for(int i = 0; i < input->categoryList->Count; i++)
	{
		CategorySummary^ temp = gcnew CategorySummary();
		temp->categoryName = input->categoryList[i]->categoryName;

		for(int j = 0; j < input->categoryList[i]->valueList->Count; j++)
		{
			

		}
	}


}*/
 
PhotoResult^ HelperClass::AnalyzePhoto(String^ SourceFileName, String^ category, String^ mosModel, bool onCloud)
{	
	PhotoResult^ photoResult= nullptr;
	FrameData *frameData = umHelper->AnalyzePhoto(	ManagedToUnmanagedString(SourceFileName), 
													ManagedToUnmanagedString(category), 
													ManagedToUnmanagedString(mosModel), 
													ManagedToUnmanagedString(this->directoryPathForFiles),
													onCloud);
	
	if(frameData != nullptr)
	{
		photoResult = gcnew PhotoResult();

		//Add Photo Details
		photoResult->PhotoDetails->Add(CreateParameter("MOS",frameData->mos, "ResultAndDetailPage"));
		photoResult->PhotoDetails->Add(CreateParameter("Resolution (Mega Pixels)",frameData->resolution, "DetailPage"));
		photoResult->PhotoDetails->Add(CreateParameter("Multi-scale Edge Acutance",frameData->final_sharpness_displayed * 100.0/67.5, "ResultAndDetailPage")); 		
		photoResult->PhotoDetails->Add(CreateParameter("Noise Signature Index",frameData->resizedImage_noise3_vector[0] * 100.0/0.306, "ResultAndDetailPage"));
		photoResult->PhotoDetails->Add(CreateParameter("Flat Region Index",frameData->resizedImage_noise2_vector[2] * 100.0/0.137, "DetailPage"));
		photoResult->PhotoDetails->Add(CreateParameter("Multi-scale Texture Acutance",frameData->multiScaleTextureAcutance*100.0/3.83, "DetailPage"));
		photoResult->PhotoDetails->Add(CreateParameter("Saturation",frameData->saturation*100.0/75.0, "ResultAndDetailPage"));
		photoResult->PhotoDetails->Add(CreateParameter("Color Warmth",frameData->colorWarmth*100.0/14586.0, "DetailPage")); 
		photoResult->PhotoDetails->Add(CreateParameter("Illumination",frameData->contrast * frameData->exposure * 100.0/18.2, "ResultAndDetailPage"));
		photoResult->PhotoDetails->Add(CreateParameter("Dynamic Range",(log(256) - frameData->dynamicRange) * 100.0/(log(256) - 0.58), "ResultAndDetailPage"));
		photoResult->PhotoDetails->Add(CreateParameter("Over Exposed (%)",frameData->percentageOverExposure, "DetailPage"));
		photoResult->PhotoDetails->Add(CreateParameter("Under Exposed (%)",frameData->percentageUnderExposure, "DetailPage"));		
		photoResult->PhotoDetails->Add(CreateParameter("Tool Version", GetBackendVersion(), "DetailPage"));

		//Add Visualizations
		photoResult->VisualizationImages->Add(CreateVisualizationImage("Sharpness Map",  gcnew String((frameData->SharpnessFile).c_str())));
		photoResult->VisualizationImages->Add(CreateVisualizationImage("RGB and Grayscale Histograms", gcnew String((frameData->DRHistogramFile).c_str())));
	}
	delete frameData;

	return photoResult;
}
