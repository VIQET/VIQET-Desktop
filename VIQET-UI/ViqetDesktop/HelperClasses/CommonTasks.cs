/*
Copyright © 2015 Intel Corporation
This program and the accompanying materials are made available under the terms of the Eclipse Public License v1.0, 
 which accompanies this distribution, and is available at http://www.eclipse.org/legal/epl-v10.html . https://github.com/viqet
* Contributors:
*    Intel Corporation - initial API and implementation and/or initial documentation
*/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace ViqetDesktop
{
    class CommonTasks
    {
        public static BitmapImage FetchImage(string fileName, int decodePixelWidth)
        {
            BitmapImage bitmapImage = new BitmapImage();
            bitmapImage.BeginInit();
            bitmapImage.DecodePixelWidth = decodePixelWidth;
            if (fileName.Contains("/Assets/")){
                bitmapImage.UriSource = new Uri(fileName, UriKind.RelativeOrAbsolute);
            }
            else
            {
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapImage.UriSource = new Uri(fileName);
            }
            bitmapImage.EndInit();
            return bitmapImage;
        }
    }
}
