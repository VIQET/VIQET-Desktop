/*
Copyright © 2015 Intel Corporation
This program and the accompanying materials are made available under the terms of the Eclipse Public License v1.0, 
 which accompanies this distribution, and is available at http://www.eclipse.org/legal/epl-v10.html . https://github.com/viqet
* Contributors:
*    Intel Corporation - initial API and implementation and/or initial documentation
*/
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using VQHelper;

namespace ViqetDesktop
{
    public class PhotoInspector
    {
        HelperClass PhotoQualityInspector = null;

        public PhotoInspector()
        {
            this.PhotoQualityInspector = new HelperClass(Constants.TempDirectory);
        }

        public void InspectAndUpdatePhoto(Photo photo, string categoryName)
        {
            //Fetch Exif Information
            photo.PhotoDetails.AddRange(ExifInspector.GetInfo(photo.SourceFilePath));
            
            try
            {
                string mosModel = Properties.Settings.Default.MOSModel;
                PhotoResult photoQualityResult = this.PhotoQualityInspector.AnalyzePhoto(photo.SourceFilePath, categoryName, Properties.Settings.Default.MOSModel, false);
                if (photoQualityResult != null)
                {
                    //Fetch the  photo details
                    foreach (VQHelper.Parameter parameter in photoQualityResult.PhotoDetails)
                    {
                        PhotoDetail photoDetail = new PhotoDetail();
                        photoDetail.ParameterName = parameter.ParameterName;
                        photoDetail.Value = parameter.Value < 0 ? Double.NaN: Math.Round(parameter.Value, 2);
                        photoDetail.ValueString = Properties.Resources.NA;
                        photoDetail.DisplayPreference = parameter.DisplayPreference;
                        photo.PhotoDetails.Add(photoDetail);
                    }

                    //Fetch visualization images
                    foreach (VQHelper.VisualizationImage parameter in photoQualityResult.VisualizationImages)
                    {
                        VisualizationImage visualizationImage = new VisualizationImage();
                        visualizationImage.Visualization = parameter.Visualization;
                        visualizationImage.FilePath = parameter.FilePath;
                        photo.VisualizationImages.Add(visualizationImage);
                    }
                }
            }
            catch (Exception exception)
            {
                MessageBox.Show(Properties.Resources.ExceptionDetails + " " + exception.ToString(), Properties.Resources.PhotoProcessingError);
            }
        }

        public string BackendVersion
        {
            get
            {
                try
                {
                    return this.PhotoQualityInspector.GetBackendVersion().ToString();
                }
                catch (Exception exception)
                {
                    MessageBox.Show(Properties.Resources.ExceptionDetails + " " + exception.ToString(), Properties.Resources.VersionFetchError);
                    return "0";
                }
            }
        }
    }
}
