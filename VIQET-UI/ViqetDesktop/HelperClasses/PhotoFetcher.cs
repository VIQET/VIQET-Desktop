using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ViqetDesktop
{
    class PhotoFetcher
    {
        public List<Photo> photoList { get; private set; }
        string resultStoreDirectory = null;//move to constants file?
        string photoDirectory = null;//move to constants file?

        public PhotoFetcher()
        {
            this.photoList = new List<Photo>();
            this.resultStoreDirectory = Path.Combine(Environment.CurrentDirectory, "ResultStore"); //Move to Constants file
            this.photoDirectory = Path.Combine(this.resultStoreDirectory, "Photos");//move to constants file?
            
            //Create the photo folder if it does not exist
            Directory.CreateDirectory(photoDirectory);
        }

        public void AddPhoto(string filepath)
        {      
            if (File.Exists(filepath))
            {
                //Copy file to data folder  
                string fileName = Path.GetFileNameWithoutExtension(filepath);
                string fileExtension = Path.GetExtension(filepath);
                string localFilePath = null;
                int appendNumber = -1;
                do
                {
                    appendNumber++;
                    localFilePath = Path.Combine(this.photoDirectory, fileName + appendNumber + fileExtension);
                }
                while(File.Exists(localFilePath));
                File.Copy(filepath,localFilePath);

                //Add file to photoList
                Photo photo = new Photo();
                photo.Filename = fileName + fileExtension;
                photo.SourceFilePath = localFilePath;
                this.photoList.Add(photo);
            }

           
        }
    }
}
