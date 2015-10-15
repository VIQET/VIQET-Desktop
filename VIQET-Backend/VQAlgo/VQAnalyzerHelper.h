// TestClrProj.h

#pragma once

#include "VideoQuality.h"

using namespace System;

namespace VQAnalyzerHelperNA {

	public ref class VQHelperClass
	{
	public:
		VQHelperClass();
		// TODO: Add your methods for this class here.
		void SetString(System::String^ str);
		System::String^ GetString();
		 VideoQuality* pGetInstance();
	private:
		System::String^ mStr;
		VideoQuality *cppcls;
	};
}
