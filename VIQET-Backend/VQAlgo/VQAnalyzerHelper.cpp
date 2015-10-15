// This is the main DLL file.
#include "VQAnalyzerHelper.h"

using namespace Runtime::InteropServices;
using namespace VQAnalyzerHelperNA;

VQHelperClass::VQHelperClass()
{
}

VideoQuality* VQHelperClass::pGetInstance()
{
	VideoQuality*vq=VideoQuality::pGetInstance();
	return vq;
}
 



void VQHelperClass::SetString(System::String^ strg)
{
	
	std::string os;
   const char* chars =(const char*)(Marshal::StringToHGlobalAnsi(strg)).ToPointer();
  // std::string chars =(std::string*)(Marshal::StringToHGlobalAnsi(strg)).ToPointer();
   os = chars;
   Marshal::FreeHGlobal(IntPtr((void*)chars));

	//cppcls->set(os);
}

System::String^ VQHelperClass::GetString()
{
	
	return mStr;
}