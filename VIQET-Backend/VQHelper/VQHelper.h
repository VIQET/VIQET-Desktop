/*
Copyright © 2015 Intel Corporation
This program and the accompanying materials are made available under the terms of the Eclipse Public License v1.0, 
 which accompanies this distribution, and is available at http://www.eclipse.org/legal/epl-v10.html . https://github.com/viqet
* Contributors:
*    Intel Corporation - initial API and implementation and/or initial documentation
*/
#include<vector>
#include <string>
#include "StdAfx.h"
#include "..\VQAlgo\FrameData.h"
#include "..\VQAlgo\UMHelper.h"

#pragma once

using namespace System;
using namespace System::Collections::Generic;

namespace VQHelper {

	public ref class Parameter
    {
		public: 
			String^ ParameterName;
			double Value;
			String^ DisplayPreference;
    };

	public ref class VisualizationImage
    {
		public:
			String^ Visualization;
			String^ FilePath;
	};

	public ref class PhotoResult
	{
		public:
			List<Parameter^>^ PhotoDetails;
			List<VisualizationImage^>^ VisualizationImages;

			PhotoResult()
			{
				PhotoDetails = gcnew List<Parameter^>();
				VisualizationImages = gcnew List<VisualizationImage^>();
			}
	};

	public ref class Summary
	{
		public:
			double mean;
			double range;
	};



	public ref class VideoFrameResult
	{
		public:
			List<Parameter^>^ VideoFrameDetails;

			VideoFrameResult()
			{
				VideoFrameDetails = gcnew List<Parameter^>();
			}
	};

	public ref class HelperClass
	{
	
	public:		
		HelperClass(String^ directoryName);	
		~HelperClass();
		int HelperClass::GetBackendVersion();
		PhotoResult^ AnalyzePhoto(String^ SourceFileName, String^ category, String^ mosModel, bool onCloud);
		Summary^ HelperClass::ValueSummary(List<double>^ values);
		List<Summary^>^ HelperClass::ValueSummaryList(List<List<double>^>^ listOfvalues);
		

	private:
		String^ directoryPathForFiles;
		array< Byte >^ byteArray;
		int fc;

		void setDirectoryPathForFiles(String^  r);
		UMHelper *umHelper;
		string ManagedToUnmanagedString(String^ s);

	};

}
