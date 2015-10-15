/*
Copyright © 2015 Intel Corporation
This program and the accompanying materials are made available under the terms of the Eclipse Public License v1.0, 
 which accompanies this distribution, and is available at http://www.eclipse.org/legal/epl-v10.html . https://github.com/viqet
* Contributors:
*    Intel Corporation - initial API and implementation and/or initial documentation
*/
using System;
using System.Collections.Generic;
using System.Collections;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ViqetDesktop
{
    [Serializable()]
    public class FeatureDetail
    {
        public string ParameterName { get; set; }
        public double Value { get; set; }
        public double StandardDeviation { get; set; }
        public double StandardError { get; set; }
        public double ConfidenceInterval95 { get; set; }
        public string DisplayPreference { get; set; }
    }

    [Serializable()]
    public class Category
    {
        public string Name { get; set; }
        public List<Photo> PhotoList { get; set; }
        public List<FeatureDetail> FeatureDetails { get { return GetDetails(); } }
        public Visibility Empty { get { return PhotoList.Count == 0 ? Visibility.Collapsed : Visibility.Visible; } }
        public Visibility NotEmpty { get { return PhotoList.Count == 0 ? Visibility.Visible : Visibility.Collapsed; } }
        private List<FeatureDetail> EmptyFeatureDetails;

        public bool isInputCategoryFollowingMethodology()
        {
                   Hashtable h = new Hashtable();
                    foreach (Photo photo in this.PhotoList)
                    {
                        if (!h.ContainsKey(photo.InputCategory))
                        {
                            h.Add(photo.InputCategory, 1);
                        }
                        else
                        {
                            int count = Int32.Parse(h[photo.InputCategory].ToString());
                            ++count;
                            h.Remove(photo.InputCategory);
                            h.Add(photo.InputCategory, count);
                        }
                    }
                    foreach (DictionaryEntry pair in h)
                    {
                        int count = Int32.Parse(h[pair.Key].ToString());
                        if (count < 5)
                        {
                            return false;
                        }
                    }
                    return true;
                
            
        }

        public List<BitmapImage> listOfStars
        {
            get
            {
                List<BitmapImage> ImageList = new List<BitmapImage>();
                if (this.PhotoList.Count < 5)
                {
                    for (int starIndex = 0; starIndex < 5; starIndex++)
                    {
                        StarImage i1 = new StarImage("/Assets/Icons/Star0.0.png");
                        ImageList.Add(i1.Thumbnail);
                    }
                }
                else
                {
                    if (!(isInputCategoryFollowingMethodology()))
                    {
                        for (int starIndex = 0; starIndex < 5; starIndex++)
                        {
                            StarImage i1 = new StarImage("/Assets/Icons/Star0.0.png");
                            ImageList.Add(i1.Thumbnail);
                        }
                    }
                    else
                    {
                        for (int starIndex = 0; starIndex < 5; starIndex++)
                        {
                            double starValue = this.MOS.Value - starIndex;

                            if (starValue > 0.9)
                            {
                                StarImage i1 = new StarImage("/Assets/Icons/Star1.0.png");
                                ImageList.Add(i1.Thumbnail);
                            }
                            else if (starValue > 0.8)
                            {
                                StarImage i1 = new StarImage("/Assets/Icons/Star0.9.png");
                                ImageList.Add(i1.Thumbnail);
                            }
                            else if (starValue > 0.7)
                            {
                                StarImage i1 = new StarImage("/Assets/Icons/Star0.8.png");
                                ImageList.Add(i1.Thumbnail);
                            }
                            else if (starValue > 0.6)
                            {
                                StarImage i1 = new StarImage("/Assets/Icons/Star0.7.png");
                                ImageList.Add(i1.Thumbnail);
                            }
                            else if (starValue > 0.5)
                            {
                                StarImage i1 = new StarImage("/Assets/Icons/Star0.6.png");
                                ImageList.Add(i1.Thumbnail);
                            }
                            else if (starValue > 0.4)
                            {
                                StarImage i1 = new StarImage("/Assets/Icons/Star0.5.png");
                                ImageList.Add(i1.Thumbnail);
                            }
                            else if (starValue > 0.3)
                            {
                                StarImage i1 = new StarImage("/Assets/Icons/Star0.4.png");
                                ImageList.Add(i1.Thumbnail);
                            }
                            else if (starValue > 0.2)
                            {
                                StarImage i1 = new StarImage("/Assets/Icons/Star0.3.png");
                                ImageList.Add(i1.Thumbnail);
                            }
                            else if (starValue > 0.1)
                            {
                                StarImage i1 = new StarImage("/Assets/Icons/Star0.2.png");
                                ImageList.Add(i1.Thumbnail);
                            }
                            else if (starValue > 0.0)
                            {
                                StarImage i1 = new StarImage("/Assets/Icons/Star0.1.png");
                                ImageList.Add(i1.Thumbnail);
                            }
                            else
                            {
                                StarImage i1 = new StarImage("/Assets/Icons/Star0.0.png");
                                ImageList.Add(i1.Thumbnail);
                            }
                        }
                    }
                    
                }
                return ImageList;
            }
        }

        public Category()
        {
            Initialize();
        }

        public Category(List<FeatureDetail> emptyFeatureDetails)
        {
            this.EmptyFeatureDetails = emptyFeatureDetails;
            Initialize();
        }

        private void Initialize()
        {
            this.PhotoList = new List<Photo>();
        }
        
        private List<FeatureDetail> GetDetails() 
        {
            List<List<double>> CategoryDetailList = new List<List<double>>();

            if (PhotoList.Count > 0)
            {
                //Use the first item in PhotoList to determine how many PhotoDetails are present
                foreach (PhotoDetail photoDetail in this.PhotoList[0].PhotoDetails)
                {
                    CategoryDetailList.Add(new List<double>());
                }

                //Extract each parameter of the photo into the categoryDetailList
                foreach (Photo photo in this.PhotoList)
                {
                    for (int parameterIndex = 0; parameterIndex < photo.PhotoDetails.Count; parameterIndex++)
                    {
                        CategoryDetailList[parameterIndex].Add(photo.PhotoDetails[parameterIndex].Value);
                    }
                }

                //Create list of aggregated parameters
                List<FeatureDetail> deviceDetails = new List<FeatureDetail>();
                for (int parameterIndex = 0; parameterIndex < this.PhotoList[0].PhotoDetails.Count; parameterIndex++)
                {
                    deviceDetails.Add(AggregateValue(this.PhotoList[0].PhotoDetails[parameterIndex].ParameterName, this.PhotoList[0].PhotoDetails[parameterIndex].DisplayPreference, CategoryDetailList[parameterIndex]));
                }
                return deviceDetails;
            }
            else
            {
                return this.EmptyFeatureDetails;
            }    
        }

        public string MOSValue
        {
            get
            {
                if(this.PhotoList.Count >= 5)
                {
                    if (isInputCategoryFollowingMethodology())
                    {
                        foreach (FeatureDetail featureDetail in GetDetails())
                        {
                            if (featureDetail.ParameterName == "MOS")
                            {
                                if (!(featureDetail.Value == 0.0))
                                {
                                    return featureDetail.Value.ToString("0.0");
                                }
                                else
                                {
                                    return Properties.Resources.NA;
                                }
                            }
                            else
                            {
                                return Properties.Resources.NA;
                            }
                        }
                    }
                }
                return Properties.Resources.NA;
            }
            
        }

        public string Variability
        {
            get
            {
                if (this.PhotoList.Count >= 5)
                {
                    if (isInputCategoryFollowingMethodology())
                    {
                        foreach (FeatureDetail featureDetail in GetDetails())
                        {
                            if (featureDetail.ParameterName == "MOS")
                            {
                                return "Variability: "+ featureDetail.StandardError.ToString("0.0");
                            }
                            else
                            {
                                return "Variability: " + Properties.Resources.NA;
                            }
                        }
                    }
                }
                return "Variability: " + Properties.Resources.NA;
            }

        }

        public FeatureDetail MOS
        {
            get
            {
                foreach (FeatureDetail featureDetail in GetDetails())
                {
                    if (featureDetail.ParameterName == "MOS")
                    {
                        return featureDetail;
                    }
                }
                return null;
            }
        }

        private FeatureDetail AggregateValue(string parameterName, string displayPreference, List<double> valueList)
        {
            //Formulas used from http://www.sjsu.edu/faculty/gerstman/StatPrimer/estimation.pdf
            int n = valueList.Count;

            //Calculate Mean
            double mean = valueList.Average();

            //Calculate Standard Deviation
            double sumSquaredDifferences = 0.0;
            foreach (double value in valueList)
            {
                double difference = value - mean;
                sumSquaredDifferences += difference * difference;
            }
            double standardDeviation = Math.Sqrt(sumSquaredDifferences / n);

            //Calculate standard error
            double standardErrorOfMean = standardDeviation / Math.Sqrt(n);

            //Calculate error for 95% confidence
            double confidenceInterval = standardErrorOfMean * TStat(n);

            FeatureDetail aggregateValue = new FeatureDetail();
            aggregateValue.ParameterName = parameterName;
            aggregateValue.Value = Math.Round(mean, 1);
            aggregateValue.StandardDeviation = Math.Round(standardDeviation,1);
            aggregateValue.StandardError = Math.Round(standardErrorOfMean,1);
            aggregateValue.ConfidenceInterval95 = Math.Round(confidenceInterval, 1);
            aggregateValue.DisplayPreference = displayPreference;
            return aggregateValue;
        }

        private double TStat(int n)
        {
            //t Table values for 95% confidence interval from http://www.sjsu.edu/faculty/gerstman/StatPrimer/t-table.pdf

            //df -> Degrees of freedom
            int df = n - 1;

            return
            df == 0 ? 0 :
            df == 1 ? 12.71 :
            df == 2 ? 4.303 :
            df == 3 ? 3.182 :
            df == 4 ? 2.776 :
            df == 5 ? 2.571 :
            df == 6 ? 2.447 :
            df == 7 ? 2.365 :
            df == 8 ? 2.306 :
            df == 9 ? 2.262 :
            df == 10 ? 2.228 :
            df == 11 ? 2.201 :
            df == 12 ? 2.179 :
            df == 13 ? 2.160 :
            df == 14 ? 2.145 :
            df == 15 ? 2.131 :
            df == 16 ? 2.120 :
            df == 17 ? 2.110 :
            df == 18 ? 2.101 :
            df == 19 ? 2.093 :
            df == 20 ? 2.086 :
            df == 21 ? 2.080 :
            df == 22 ? 2.074 :
            df == 23 ? 2.069 :
            df == 24 ? 2.064 :
            df == 25 ? 2.060 :
            df == 26 ? 2.056 :
            df == 27 ? 2.052 :
            df == 28 ? 2.048 :
            df == 29 ? 2.045 :
            df < 40 ? 2.042 :
            df < 60 ? 2.021 :
            df < 80 ? 2.000 :
            df < 100 ? 1.990 :
            df < 1000 ? 1.984 :
            1.962;
        }
    }
}
