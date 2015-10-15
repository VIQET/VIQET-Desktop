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
using System.Xml;

namespace ViqetDesktop
{
    class XmlFileWriter
    {
        public static void CreateFile(Result result, string filePath)
        {
            using (StreamWriter outFile = new StreamWriter(filePath))
            {
                XmlWriterSettings settings = new XmlWriterSettings();
                settings.Async = false;
                settings.Indent = true;
                settings.IndentChars = "\t";
                settings.NewLineOnAttributes = true;

                using (XmlWriter writer = XmlWriter.Create(outFile, settings))
                {
                    //Write the top level Result info
                    writer.WriteStartElement("Result");
                    writer.WriteAttributeString("DeviceName", result.Name);
                    writer.WriteAttributeString("PhotoCount", result.PhotoCount().ToString());
                    
                        //Overall Score
                        FeatureDetail overallMOSDetail = result.GetOverallMOS();
                        writer.WriteStartElement("OverallScore");
                            writer.WriteStartElement("Property");
                            writer.WriteAttributeString("Name", overallMOSDetail.ParameterName);
                            writer.WriteAttributeString("Value", overallMOSDetail.Value.ToString());
                            writer.WriteAttributeString("StandardError", overallMOSDetail.StandardError.ToString());
                            writer.WriteEndElement();
                        writer.WriteEndElement();

                        //Categories
                        writer.WriteStartElement("Categories");
                        foreach (Category category in result.outputCategoryList)
                        {
                            writer.WriteStartElement("Category");
                            writer.WriteAttributeString("Name", category.Name);
                            writer.WriteAttributeString("Value", category.PhotoList.Count.ToString());

                                //Category Score
                                writer.WriteStartElement("Categories");
                                foreach (FeatureDetail featureDetail in category.FeatureDetails)
                                {
                                    writer.WriteStartElement("Property");
                                    writer.WriteAttributeString("Name", featureDetail.ParameterName);
                                    if (category.MOSValue.Equals(Properties.Resources.NA))
                                    {
                                        writer.WriteAttributeString("Value", Properties.Resources.NA);
                                    }
                                    else
                                    {
                                        writer.WriteAttributeString("Value", featureDetail.Value.ToString());
                                    }                                    
                                    writer.WriteAttributeString("StandardError", featureDetail.StandardError.ToString());
                                    writer.WriteEndElement();
                                }
                                writer.WriteEndElement();


                                //Photos
                                writer.WriteStartElement("Photos");
                                foreach (Photo photo in category.PhotoList)
                                {
                                    writer.WriteStartElement("Photo");
                                    writer.WriteAttributeString("Filename", photo.Filename);
                                    writer.WriteAttributeString("Filepath", photo.SourceFilePath);
                                    
                                    //Photo Details
                                    foreach (PhotoDetail photoDetail in photo.PhotoDetails)
                                    {
                                        if (photoDetail.DisplayPreference == Constants.DisplayPreference.DETAIL_PAGE || photoDetail.DisplayPreference == Constants.DisplayPreference.RESULT_AND_DETAIL_PAGE)
                                        {
                                            writer.WriteStartElement("PhotoDetail");
                                            writer.WriteAttributeString("Name", photoDetail.ParameterName);
                                            writer.WriteAttributeString("Value", photoDetail.Value.ToString());
                                            writer.WriteEndElement();
                                        }
                                    }
                                    writer.WriteEndElement();
                                }
                                writer.WriteEndElement();

                            writer.WriteEndElement();
                        }
                        writer.WriteEndElement();

                    writer.WriteEndElement();
                }
            }
        }
    }
}
