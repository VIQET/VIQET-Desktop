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
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;


namespace ViqetDesktop
{
    class ResultStore
    {
        public static List<Result> GetResultList()
        {
            List<Result> resultList = new List<Result>();

            if (Directory.Exists(Constants.PersistedResultsFileDirectory))
            {
                string[] persistedResultFiles = Directory.GetFiles(Constants.PersistedResultsFileDirectory, "*.vqt");

                foreach (string persistedResultFile in persistedResultFiles)
                {
                    if (File.Exists(persistedResultFile))
                    {
                        Stream ResultFileStream = File.OpenRead(persistedResultFile);
                        BinaryFormatter deserializer = new BinaryFormatter();
                        Result result = (Result)deserializer.Deserialize(ResultFileStream);
                        ResultFileStream.Close();

                        resultList.Add(result);
                    }
                }
            }

            return resultList;
        }

        public static void SaveResult(Result result)
        {
            //Save the source images of each photo
            foreach (Category category in result.outputCategoryList)
            {
                foreach (Photo photo in category.PhotoList)
                {
                    if (!photo.Persisted)
                    {
                        //Save the visualization images
                        foreach (VisualizationImage visualizationImage in photo.VisualizationImages)
                        {
                            visualizationImage.PersistSources();
                        }

                        //Save the photo
                        photo.PersistSources();

                        photo.Persisted = true;
                    }
                }
            }
            
            //Find a unique name for the result file if it is a new result
            if (result.FilePath == null)
            {
                string resultFilePath = null;
                int appendNumber = -1;
                do
                {
                    appendNumber++;
                    resultFilePath = Path.Combine(Constants.PersistedResultsFileDirectory, result.Name + appendNumber + ".vqt");
                }
                while (File.Exists(resultFilePath));

                result.FilePath = resultFilePath;
            }

            //Create the file and persist the result
            Stream ResultFileStream = File.Create(result.FilePath);
            BinaryFormatter serializer = new BinaryFormatter();
            serializer.Serialize(ResultFileStream, result);
            ResultFileStream.Close();
        }

        public static void DeleteResult(Result result)
        {
            foreach (Photo photo in result.PhotoList)
            {
                //Delete the source photo
                if (File.Exists(photo.SourceFilePath))
                {
                    File.Delete(photo.SourceFilePath);
                }

                //Delete the visualization images
                foreach (VisualizationImage visualizationImage in photo.VisualizationImages)
                {
                    if (File.Exists(visualizationImage.FilePath))
                    {
                        File.Delete(visualizationImage.FilePath);
                    }
                }
            }

            //Delete the persisted result file from disk
            if (result.FilePath != null && File.Exists(result.FilePath))
            {
                File.Delete(result.FilePath);
            }
        }

        public static void DeleteAllResults()
        {
            Directory.Delete(Constants.ResultStoreDirectory, true);
            Directory.Delete(Constants.TempDirectory, true);
        }
    }
}
