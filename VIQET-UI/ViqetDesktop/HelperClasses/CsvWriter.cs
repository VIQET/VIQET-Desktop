/*
Copyright © 2015 Intel Corporation
This program and the accompanying materials are made available under the terms of the Eclipse Public License v1.0, 
 which accompanies this distribution, and is available at http://www.eclipse.org/legal/epl-v10.html . https://github.com/viqet
* Contributors:
*    Intel Corporation - initial API and implementation and/or initial documentation
*/
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ViqetDesktop
{
    class CsvWriter
    {
        public static void CreateFile(Result result, string filePath)
        {
            using (StreamWriter outFile = new StreamWriter(filePath))
            {
                //Write the overall score
                FeatureDetail overallMOSDetail = result.GetOverallMOS();
                string overallMOSHeading = "Overall " + overallMOSDetail.ParameterName + ",Standard Error ";
                string overallMOS = overallMOSDetail.Value.ToString("0.0") + "," + overallMOSDetail.StandardError.ToString("0.0");
                outFile.WriteLine(overallMOSHeading);
                outFile.WriteLine(overallMOS);

                outFile.WriteLine("");
                outFile.WriteLine("");
                
                //Heading for summary section
                string summaryHeading = "Category Name";
                foreach (FeatureDetail featureDetail in result.outputCategoryList.ElementAt(0).FeatureDetails)
                {
                    summaryHeading += "," + featureDetail.ParameterName + ",Standard Error";
                }
                outFile.WriteLine(summaryHeading);

                //Write a summary for each category
                foreach (Category category in result.outputCategoryList)
                {
                    string categorySummary = category.Name;
                    foreach (FeatureDetail featureDetail in category.FeatureDetails)
                    {
                        if (category.MOSValue.Equals(Properties.Resources.NA))
                        {
                            categorySummary += "," + Properties.Resources.NA + "," + Properties.Resources.NA;
                        }
                        else
                        {
                            categorySummary += "," + featureDetail.Value + "," + featureDetail.StandardError;
                        }
                    }
                    outFile.WriteLine(categorySummary);
                }


                outFile.WriteLine("");
                outFile.WriteLine("");

                //Photo Detal Section
                if (result.outputCategoryList.Count > 0)
                {
                    //Use the first photo of the first category to populate the heading for the photo detail section (Data for each file)
                    string commaSeparatedPhotoParameterNames = string.Empty;
                    foreach (PhotoDetail photoDetail in result.outputCategoryList[0].PhotoList[0].PhotoDetails)
                    {
                            commaSeparatedPhotoParameterNames += "," + photoDetail.ParameterName;
                    }
                    commaSeparatedPhotoParameterNames += "," + "Category";
                    outFile.WriteLine("Filename" + commaSeparatedPhotoParameterNames);

                    foreach (Category category in result.outputCategoryList)
                    {
                        if (category.PhotoList.Count > 0)
                        {
                            //Write Each File Parameter
                            foreach (Photo photo in category.PhotoList)
                            {
                                string commaSeparatedPhotoDetails = string.Empty;
                                foreach (PhotoDetail photoDetail in photo.PhotoDetails)
                                {
                                        commaSeparatedPhotoDetails += "," + photoDetail.Value;
                                }
                                commaSeparatedPhotoDetails += "," + category.Name;

                                outFile.WriteLine(photo.Filename + commaSeparatedPhotoDetails);
                            }
                        }
                    } 
                }
            }
        }

        public static void CreateFileForAutomationLab(Result result, string filePath)
        {
            using (StreamWriter outFile = new StreamWriter(filePath))
            {
                //Header Line
                outFile.WriteLine("Metric,Value");


                //Overall Section - Write the overall score
                FeatureDetail overallMOSDetail = result.GetOverallMOS();
                outFile.WriteLine("Overall - " + overallMOSDetail.ParameterName + "," + overallMOSDetail.Value + " +/- " + overallMOSDetail.StandardError);

                //Each Category
                foreach (Category category in result.outputCategoryList)
                {
                    foreach (FeatureDetail featureDetail in category.FeatureDetails)
                    {
                        outFile.WriteLine( category.Name + " - Aggregate - " + featureDetail.ParameterName + "," + featureDetail.Value + " +/- " + featureDetail.StandardError);
                    }
                }

                //Each Photo
                foreach (Category category in result.outputCategoryList)
                {
                    foreach (Photo photo in category.PhotoList)
                    {
                        foreach (PhotoDetail photoDetail in photo.PhotoDetails)
                        {
                            outFile.WriteLine(category.Name + " - " + photo.Filename + " - " + photoDetail.ParameterName + "," + photoDetail.Value);
                        }
                    }
                }
            }
        }
    }
}
