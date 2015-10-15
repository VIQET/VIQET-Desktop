/*
Copyright © 2015 Intel Corporation
This program and the accompanying materials are made available under the terms of the Eclipse Public License v1.0, 
 which accompanies this distribution, and is available at http://www.eclipse.org/legal/epl-v10.html . https://github.com/viqet
* Contributors:
*    Intel Corporation - initial API and implementation and/or initial documentation
*/
using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ViqetDesktop
{
    class ResultsExporter
    {
        public static void ExportToCSV(Result result)
        {
            SaveFileDialog saveFileDialog = new SaveFileDialog();
            saveFileDialog.FileName = result.Name + " v" + Constants.Version;
            saveFileDialog.DefaultExt = ".csv";
            saveFileDialog.Filter = "Comma Separated Value File (.csv)|*.csv";

            Nullable<bool> fileNameCreationSuccess = saveFileDialog.ShowDialog();
            if (fileNameCreationSuccess == true)
            {
                CsvWriter.CreateFile(result, saveFileDialog.FileName);
                CsvWriter.CreateFileForAutomationLab(result, saveFileDialog.FileName + " Automation.csv");

                //Automatically launch the generated CSV file
                Process.Start(saveFileDialog.FileName);
            }
        }

        public static void ExportToXML(Result result)
        {
            SaveFileDialog saveFileDialog = new SaveFileDialog();
            saveFileDialog.FileName = result.Name + " v" + Constants.Version;
            saveFileDialog.DefaultExt = ".xml";
            saveFileDialog.Filter = "XML File (.xml)|*.xml";

            Nullable<bool> fileNameCreationSuccess = saveFileDialog.ShowDialog();
            if (fileNameCreationSuccess == true)
            {
                XmlFileWriter.CreateFile(result, saveFileDialog.FileName);

                //Automatically launch the generated XML file
                Process.Start(saveFileDialog.FileName);
            }
        }

        public static void ExportToPDF(Result result)
        {
            SaveFileDialog saveFileDialog = new SaveFileDialog();
            saveFileDialog.FileName = result.Name + " v" + Constants.Version;
            saveFileDialog.DefaultExt = ".pdf";
            saveFileDialog.Filter = "PDF File (.pdf)|*.pdf";

            Nullable<bool> fileNameCreationSuccess = saveFileDialog.ShowDialog();
            if (fileNameCreationSuccess == true)
            {
                PdfWriter.CreateFile(result, saveFileDialog.FileName);

                //Automatically launch the generated PDF file
                Process.Start(saveFileDialog.FileName);
            }
        }
    }
}
