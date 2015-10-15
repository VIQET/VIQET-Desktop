/*
Copyright © 2015 Intel Corporation
This program and the accompanying materials are made available under the terms of the Eclipse Public License v1.0, 
 which accompanies this distribution, and is available at http://www.eclipse.org/legal/epl-v10.html . https://github.com/viqet
* Contributors:
*    Intel Corporation - initial API and implementation and/or initial documentation
*/
using namespace std;
#include <string>

#pragma once
class Constants
{
public:
	
	//Input Categories
	static const string LANDSCAPE;
	static const string DINNER_PLATE;
	static const string WALL_HANGING;
	static const string NIGHT_SHOT;

	//MOS Mappings
	static const string CategoryMOS_RSvm;
	static const string GenericMOS_RSvm;
	static const string CategoryMOS_LibSvm;
	static const string GenericMOS_LibSvm;

	static const double colorWarmth[391][3];
};

