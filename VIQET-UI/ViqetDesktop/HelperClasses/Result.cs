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
using System.Threading.Tasks;
using System.Windows;

namespace ViqetDesktop
{
    [Serializable()]
    public class Result
    {
        public string Name { get; set; }
        public string FilePath { get; set; }
        public List<Photo> PhotoList { get; set; }
        public Methodology Methodology { get; private set; }

        public List<Category> InputCategoryList 
        { 
            get 
            {
                Dictionary<string, Category> inputCategoryDictionary = new Dictionary<string, Category>();

                //Add categories from methodology
                foreach (string inputCategoryName in this.Methodology.InputCategories.Keys)
                {
                    Category category = new Category();
                    category.Name = inputCategoryName;
                    category.PhotoList = new List<Photo>();
                    inputCategoryDictionary.Add(category.Name, category);
                }

                //Divide the photos into input categories                
                foreach (Photo photo in this.PhotoList)
                {
                    Category category = inputCategoryDictionary[photo.InputCategory];
                    
                    //Add the photo to the category
                    category.PhotoList.Add(photo);
                }

                return inputCategoryDictionary.Values.ToList();
            }
        }
        public List<Category> outputCategoryList 
        {
            get
            {
                //Use the first photo to create emptyFeatureDetails
                List<FeatureDetail> emptyFeatureDetails = new List<FeatureDetail>();
                foreach (PhotoDetail photodetail in this.PhotoList.ElementAt(0).PhotoDetails)
                {
                        FeatureDetail featureDetail = new FeatureDetail();
                        featureDetail.ParameterName = photodetail.ParameterName;
                        featureDetail.DisplayPreference = photodetail.DisplayPreference;
                        featureDetail.ConfidenceInterval95 = 0.0;
                        featureDetail.StandardDeviation = 0.0;
                        featureDetail.StandardError = 0.0;
                        featureDetail.Value = 0.0;
                        emptyFeatureDetails.Add(featureDetail);
                }
                
                //Make a Dictionary of OutputCategories
                Dictionary<string, Category> outputCategoryDictionary = new Dictionary<string, Category>();
                foreach (string categoryName in this.Methodology.OutputCategories.Keys)
                {
                    Category category = new Category(emptyFeatureDetails);
                    category.Name = categoryName;
                    category.PhotoList = new List<Photo>();
                    outputCategoryDictionary.Add(category.Name, category);
                }

                //Divide the photos into output categories
                foreach (Photo photo in this.PhotoList)
                {
                    //Each photo can be in more than one output category
                    foreach (string categoryName in photo.OutputCategories)
                    {

                        //if this is the first time this category is seen, add it to the dictionary
                        if (!outputCategoryDictionary.ContainsKey(categoryName))
                        {
                            MessageBox.Show("Dll returned an output category not specified in the methodology");                        //    category = new Category();
                        }
                        
                        Category category = outputCategoryDictionary[categoryName];

                        //Add the photo to the category
                        category.PhotoList.Add(photo);
                    } 
                }
                return outputCategoryDictionary.Values.ToList();
            } 
        }

        public Result()
        {
            this.Name = Properties.Resources.DefaultTestName;
            this.PhotoList = new List<Photo>();
            this.Methodology = MethodologyProvider.Get().Methodology;            
        }

        public bool AllAnalyzed()
        {
            foreach (Photo photo in this.PhotoList)
            {
                if(!photo.Analyzed)
                {
                    return false;
                }
            }
            return true;
        }

        public bool MethodologyFollowed
        {
            get
            {
                foreach (Photo photo in this.PhotoList)
                {
                    foreach (PhotoDetail photoDetail in photo.PhotoDetails)
                    {
                        if (double.IsNaN(photoDetail.Value))
                        {
                            return false;
                        }
                    }
                }
                return true;
            }
        }

        /// <summary>
        /// Take the mean and std deviation and generate many numbers
        /// </summary>
        /// <param name="mean"></param>
        /// <param name="standardError"></param>
        /// <param name="bootstrapCount"></param>
        /// <returns></returns>
        private List<double> BootStrap(double mean, double standardError, int bootstrapCount)
        {
            Random random = new Random();
            List<double> bootstrappedValues = new List<double>(bootstrapCount);

            for (int index = 0; index < bootstrapCount; index++)
            {
                double uniform1 = random.NextDouble();
                double uniform2 = random.NextDouble();
                double randomStdNormal = Math.Sqrt(-2.0 * Math.Log(uniform1)) * Math.Sin(2.0 * Math.PI * uniform2);
                double randNormal = mean + standardError * randomStdNormal;
                bootstrappedValues.Add(randNormal);
            }

            return bootstrappedValues;
        }

        /// <summary>
        /// Take a list of Feature Details, bootstrap and return a List of averages
        /// </summary>
        /// <param name="featureDetailList"></param>
        /// <returns></returns>
        private List<double> GenerateAverageList(List<FeatureDetail> featureDetailList, int bootstrapCount)
        {
            List<double> averageList = new List<double>(featureDetailList.Count * bootstrapCount);
            foreach(FeatureDetail featureDetail in featureDetailList)
            {
                averageList.AddRange(BootStrap(featureDetail.Value, featureDetail.StandardError, bootstrapCount));
            }
            return averageList;
        }

        public FeatureDetail GetOverallMOS()
        {
            //Check if any category is empty or Does not have 5 Photos
            foreach (Category category in this.outputCategoryList)
            {
                if (category.MOSValue.Equals(Properties.Resources.NA))
                {
                    FeatureDetail incompleteMOS = new FeatureDetail();

                    //We use the first element for the Parameter name and display preference
                    incompleteMOS.ParameterName = "MOS";
                    incompleteMOS.Value = double.NaN;
                    incompleteMOS.StandardDeviation = Math.Round(0.0, 1);
                    incompleteMOS.StandardError = Math.Round(0.0, 1);
                    incompleteMOS.ConfidenceInterval95 = Math.Round(0.0, 1);

                    return incompleteMOS;
                }
            }

            int lcm = 1;
            foreach (Category category in this.outputCategoryList)
            {
                lcm = LCM(lcm, category.PhotoList.Count);
            }
            
            //Create a mosList of items where we have equal number of items for each category
            List<double> mosList = new List<double>();
            foreach (Category category in this.outputCategoryList)
            {
                int copyCount = lcm / category.PhotoList.Count;

                foreach (Photo photo in category.PhotoList)
                {
                    foreach (PhotoDetail photoDetail in photo.PhotoDetails)
                    {
                        if (photoDetail.ParameterName == "MOS")
                        {
                            for (int copies = 0; copies < copyCount; copies++)
                            {
                                mosList.Add(photoDetail.Value);
                            }
                        }
                    }
                }
            }

            //Calculate mean
            double mean = mosList.Average();

            //Calculate Standard Deviation
            double sumSquaredDifferences = 0.0;
            foreach (double value in mosList)
            {
                double difference = value - mean;
                sumSquaredDifferences += difference * difference;
            }
            double standardDeviation = Math.Sqrt(sumSquaredDifferences / mosList.Count);

            //Calculate standard error
            double standardErrorOfMean = standardDeviation / Math.Sqrt(mosList.Count);

            FeatureDetail aggregateMOS = new FeatureDetail();

            //We use the first element for the Parameter name and display preference
            aggregateMOS.ParameterName = "MOS";
            aggregateMOS.Value = Math.Round(mean, 1);
            aggregateMOS.StandardDeviation = Math.Round(standardDeviation, 1);
            aggregateMOS.StandardError = Math.Round(standardErrorOfMean, 1);
            aggregateMOS.ConfidenceInterval95 = Math.Round(1.96 * standardDeviation, 1);

            return aggregateMOS;
        }


        public int PhotoCount()
        {
            int photoCount = 0;
            foreach (Category category in this.outputCategoryList)
            {
                photoCount += category.PhotoList.Count;
            }
            return photoCount;
        }

        //http://rosettacode.org/wiki/Greatest_common_divisor
        private int GCD(int a, int b)
        {
            int t;

            // Ensure B > A
            if (a > b)
            {
                t = b;
                b = a;
                a = t;
            }

            // Find GCD
            while (b != 0)
            {
                t = a % b;
                a = b;
                b = t;
            }

            if (a > 0)
            {
                return a;
            }
            else
            {
                return 1;
            }
        }

        private int LCM(int a, int b)
        {
            return a * b / GCD(a, b);
        }
    }
}
