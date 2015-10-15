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
using System.Windows.Media.Imaging;

namespace ViqetDesktop
{
    [Serializable()]
    public class PhotoDetail
    {
        public string ParameterName { get; set; }
        public double Value { get; set; }
        public string ValueString { get; set; }
        public string DisplayPreference { get; set; }
    }

    [Serializable()]
    public class VisualizationImage
    {
        public string Visualization { get; set; }
        public string FilePath { get; set; }

        public void PersistSources()
        {
            string tempFolderFilePath = this.FilePath;
            if (File.Exists(tempFolderFilePath))
            {
                //Copy the photo to local storage                
                string fileName = Path.GetFileNameWithoutExtension(tempFolderFilePath);
                string fileExtension = Path.GetExtension(tempFolderFilePath);
                string localFilePath = null;
                int appendNumber = -1;
                do
                {
                    appendNumber++;
                    localFilePath = Path.Combine(Constants.PhotoDirectory, fileName + appendNumber + fileExtension);
                }
                while (File.Exists(localFilePath));
                File.Copy(tempFolderFilePath, localFilePath);

                //Update photo with new filename and path
                this.FilePath = localFilePath; //Save the local store path to SourceFilePath

                //Delete Visualization Image from temp folder
                File.Delete(tempFolderFilePath);
            }
        }
    }

    [Serializable()]
    public class Photo
    {
        public string Name { get; set; }
        public string Filename { get; set; }
        public string SourceFilePath { get; set; }
        public string InputCategory { get; set; }
        public List<string> OutputCategories { get; set; }
        public BitmapImage Thumbnail
        {
            get { return CommonTasks.FetchImage(this.SourceFilePath, 100); }
        }
        public bool Analyzed
        {
            get 
            {
                if (this.PhotoDetails == null || this.PhotoDetails.Count == 0)
                    return false;
                else
                    return true;
            }
        }

        public List<PhotoDetail> PhotoDetails { get; set; }
        public List<VisualizationImage> VisualizationImages { get; set; }
        public bool Persisted { get; set; }

        public Photo()
        {
            Name = string.Empty;
            Filename = string.Empty;
            SourceFilePath = string.Empty;
            Persisted = false;

            PhotoDetails = new List<PhotoDetail>();
            VisualizationImages = new List<VisualizationImage>();
        }

        public void PersistSources()
        {
            if (File.Exists(this.SourceFilePath))
            {
                //Copy the photo to local storage                
                string fileName = Path.GetFileNameWithoutExtension(this.SourceFilePath);
                string fileExtension = Path.GetExtension(this.SourceFilePath);
                string localFilePath = null;
                int appendNumber = -1;
                do
                {
                    appendNumber++;
                    localFilePath = Path.Combine(Constants.PhotoDirectory, fileName + appendNumber + fileExtension);
                }
                while (File.Exists(localFilePath));
                File.Copy(this.SourceFilePath, localFilePath);

                //Update photo with new filename and path
                this.Filename = fileName + fileExtension;
                this.SourceFilePath = localFilePath; //Save the local store path to SourceFilePath
            }
        }
    }
}
