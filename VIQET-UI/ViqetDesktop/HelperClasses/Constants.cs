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
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace ViqetDesktop
{
    public static class Constants
    {
        private const string AppVersion = "2";

        private const string UIVersion = "87";

        public static string TempDirectory
        {
            get { return GetFolderFromCurrentDirectory("Temp"); }
        }

        public static string Version
        {
            get
            {              
                //OverallVersion.Methodology.Backend.Gui
                PhotoInspector photoInspector = new PhotoInspector();
                return AppVersion + "." + MethodologyProvider.Get().Methodology.Version + "." + photoInspector.BackendVersion + "." + UIVersion;
            }
        }

        public static string ResultStoreDirectory
        {
            get { return GetFolderFromCurrentDirectory("ResultStore"); }
        }

        public static string LocalStorageFile
        {
            get
            {
                string app_name = "VIQET";
                string folderBase = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string dir = string.Format(@"{0}\{1}\", folderBase, app_name);

                return Path.Combine(dir, "results.bin");
            }
        }

        public static string PhotoDirectory
        {
            get
            {
                string app_name = "VIQET";
                string folderBase = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string dir = string.Format(@"{0}\{1}\", folderBase, app_name);
                string photoDirectory = Path.Combine(dir, "Photos");

                //string photoDirectory = Path.Combine(ResultStoreDirectory, "Photos");
                if (!Directory.Exists(photoDirectory)) Directory.CreateDirectory(photoDirectory);
                return photoDirectory;
            }
        }

        public static string PersistedResultsFileDirectory
        {
            get
            {
                string app_name = "VIQET";
                string folderBase = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string dir = string.Format(@"{0}\{1}\", folderBase, app_name);
                string persistedResultsDirectory = Path.Combine(dir, "Results");
                
                //string persistedResultsDirectory = Path.Combine(ResultStoreDirectory, "Results");
                if (!Directory.Exists(persistedResultsDirectory)) Directory.CreateDirectory(persistedResultsDirectory);
                return persistedResultsDirectory;
            }
        }


        private static string GetFolderFromCurrentDirectory(string folderName)
        {
            string app_name = "VIQET";
            string folderBase = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string dir = string.Format(@"{0}\{1}\", folderBase, app_name);
            string folder = Path.Combine(dir, folderName);

            //string folder = Path.Combine(Environment.CurrentDirectory, folderName);
            if (!Directory.Exists(folder)) Directory.CreateDirectory(folder);
            return folder;
        }

        public struct DisplayPreference
        {
            public const string RESULT_PAGE = "ResultPage";
            public const string DETAIL_PAGE = "DetailPage";
            public const string RESULT_AND_DETAIL_PAGE = "ResultAndDetailPage";
            public const string NONE = "None";
        }

        public struct InputCategoryNames
        {
            public const string LANDSCAPE = "Landscape";
            public const string STILL_LIFE = "Still Life";
            public const string WALL_HANGING = "Wall Hanging";
            public const string NIGHT = "Night";
            public const string OTHER = "Other";
        }

        public struct OutputCategoryNames
        {
            public const string OUTDOOR_DAY = "Outdoor Day";
            public const string OUTDOOR_NIGHT = "Outdoor Night";
            public const string INDOOR = "Indoor";
            public const string OTHER = "Other";
        }
    }
}
