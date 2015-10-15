/*
Copyright © 2015 Intel Corporation
This program and the accompanying materials are made available under the terms of the Eclipse Public License v1.0, 
 which accompanies this distribution, and is available at http://www.eclipse.org/legal/epl-v10.html . https://github.com/viqet
* Contributors:
*    Intel Corporation - initial API and implementation and/or initial documentation
*/
using MigraDoc.DocumentObjectModel;
using MigraDoc.DocumentObjectModel.Shapes;
using MigraDoc.DocumentObjectModel.Tables;
using MigraDoc.Rendering;
using PdfSharp.Drawing;
using PdfSharp.Pdf;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace ViqetDesktop
{
    /// <summary>
    /// Using dlls from the following open source projects to produce the Pdf file 
    /// http://www.pdfsharp.com
    /// http://www.migradoc.com
    /// http://sourceforge.net/projects/pdfsharp
    /// </summary>
    class PdfWriter
    {
        public static void CreateFile(Result result, string filePath)
        {
            // Create a MigraDoc document
            Document document = CreateDocument(result);
            document.UseCmykColor = true;

            // A flag indicating whether to create a Unicode PDF or a WinAnsi PDF file.
            // This setting applies to all fonts used in the PDF document.
            // This setting has no effect on the RTF renderer.
            const bool unicode = false;

            // An enum indicating whether to embed fonts or not.
            // This setting applies to all font programs used in the document.
            // This setting has no effect on the RTF renderer.
            // (The term 'font program' is used by Adobe for a file containing a font. Technically a 'font file'
            // is a collection of small programs and each program renders the glyph of a character when executed.
            // Using a font in PDFsharp may lead to the embedding of one or more font programms, because each outline
            // (regular, bold, italic, bold+italic, ...) has its own fontprogram)
            const PdfFontEmbedding embedding = PdfFontEmbedding.Always;

            // Create a renderer for the MigraDoc document.
            PdfDocumentRenderer pdfRenderer = new PdfDocumentRenderer(unicode, embedding);
            pdfRenderer.Document = document;
            pdfRenderer.RenderDocument();
            pdfRenderer.PdfDocument.Save(filePath);
        }

        class StyleLibrary
        {
            public static Font RedTitleFont
            {
                get
                {
                    Font font = new Font();
                    font.Bold = true;
                    font.Size = 30;
                    font.Color = Color.FromCmyk(0, 255, 255, 0); ;
                    return font;
                }
            }

            public static Font BlackTitleFont
            {
                get
                {
                    Font font = new Font();
                    font.Bold = true;
                    font.Size = 30;
                    font.Color = Color.FromCmyk(100, 140, 23, 255);
                    return font;
                }
            }

            public static Font HeadingFont
            {
                get
                {
                    Font font = new Font();
                    font.Bold = true;
                    font.Color = Color.FromCmyk(100, 140, 23, 255);
                    font.Size = 18;
                    return font;
                }
            }

            public static Font TextFont
            {
                get
                {
                    Font font = new Font();
                    font.Color = Color.FromCmyk(100, 140, 23, 255);
                    font.Size = 10;
                    return font;
                }
            }

            public static Font TextFontBold
            {
                get
                {
                    Font font = new Font();
                    font.Bold = true;
                    font.Color = Color.FromCmyk(100, 140, 23, 255);
                    font.Size = 10;
                    return font;
                }
            }

            public static Font TableFont
            {
                get
                {
                    Font font = new Font();
                    font.Color = Color.FromCmyk(100, 140, 23, 255);
                    font.Size = 8;
                    return font;
                }
            }

            public static Font TableFontBold
            {
                get
                {
                    Font font = new Font();
                    font.Bold = true;
                    font.Color = Color.FromCmyk(100, 140, 23, 255);
                    font.Size = 8;
                    return font;
                }
            }
        }

        static string CreateFileFromResource(string filePath)
        {
            string filename = Path.GetFileName(filePath);
            
            BitmapImage image = new BitmapImage();
            image.BeginInit();
            image.UriSource = new Uri("pack://application:,,,/ViqetDesktop;component" + filePath);
            image.EndInit();

            PngBitmapEncoder pngEncoder = new PngBitmapEncoder();
            pngEncoder.Frames.Add(BitmapFrame.Create(image));
            using (FileStream fs = new FileStream(filename, FileMode.Create))
            {
                pngEncoder.Save(fs);
            }

            return filename;
        }

        static Document CreateDocument(Result result)
        {           
            // Create a new MigraDoc document
            Document document = new Document();

            #region Introduction Section

            Section introductionSection = document.AddSection();
                        
            // Page Title
            Paragraph titleParagraph = introductionSection.AddParagraph();
            titleParagraph.Format.Alignment = ParagraphAlignment.Center;
            titleParagraph.AddFormattedText(Properties.Resources.ProductDescription, StyleLibrary.BlackTitleFont);
            titleParagraph.AddFormattedText(" (" + Properties.Resources.ProductName + ")", StyleLibrary.BlackTitleFont);

            // Introductory paragraph Title
            Paragraph introParagraphTitle = introductionSection.AddParagraph();
            introParagraphTitle.AddLineBreak();
            introParagraphTitle.AddLineBreak();
            introParagraphTitle.AddFormattedText("Introduction", StyleLibrary.HeadingFont);

            // Introductory paragraph 
            Paragraph introParagraph = introductionSection.AddParagraph();
            introParagraph.AddLineBreak();
            introParagraph.AddFormattedText("VIQET is an easy to use no-reference photo quality evaluation tool. In order to perform photo quality evaluation, VIQET requires a set of photos "+
                                            "from a test device. It computes an overall image MOS score for a device based on the individual computed photo qualities of each image in the set. "+
                                            "The overall MOS is computed based on a number of features extracted from each image. The mapping from extracted features to MOS is based on a "+
                                            "psychophysics study that was conducted to create a large dataset of photos and associated subjective mean opinion ratings. The study was used to learn "+
                                            "a mapping from features to scores. The predicted quality score falls in the range of 1 to 5, where 1 corresponds to a low quality score and 5 corresponds to excellent quality", StyleLibrary.TextFont);

            Paragraph overallParagraphTitle = introductionSection.AddParagraph();
            overallParagraphTitle.AddLineBreak();
            overallParagraphTitle.AddLineBreak();
            overallParagraphTitle.AddFormattedText("Test Results", StyleLibrary.HeadingFont);

            // Overall Scores
            Paragraph overallScoreTitle = introductionSection.AddParagraph();
            overallScoreTitle.AddLineBreak();
            overallScoreTitle.AddFormattedText("Device Name: " + result.Name, StyleLibrary.TextFont);
            overallScoreTitle.AddLineBreak();
            overallScoreTitle.AddFormattedText("Total number of photos: " + result.PhotoList.Count, StyleLibrary.TextFont);
            overallScoreTitle.AddLineBreak();
            overallScoreTitle.AddFormattedText("Tool Version: " + Properties.Resources.VersionName + " " + Constants.Version, StyleLibrary.TextFont);

            FeatureDetail overallMOSDetail = result.GetOverallMOS();
            string overallMOS = double.IsNaN(overallMOSDetail.Value) ? "N/A" : overallMOSDetail.Value.ToString("0.0") + " +/-" + overallMOSDetail.StandardError.ToString("0.0");
            overallScoreTitle.AddLineBreak();
            overallScoreTitle.AddLineBreak();
            overallScoreTitle.AddFormattedText("Overall Score: " + overallMOS, StyleLibrary.HeadingFont);

            #endregion

            #region Summary for Categories

            // Categories Summary Section 
            Section categoriesSummarySection = document.AddSection();

            // Introductory paragraph
            Paragraph categoriesSummaryParagraphTitle = categoriesSummarySection.AddParagraph();
            categoriesSummaryParagraphTitle.AddFormattedText("Photo Category Results", StyleLibrary.HeadingFont);
            

            foreach (Category outputCategory in result.outputCategoryList)
            {
                Paragraph categoriesSummaryParagraph = categoriesSummarySection.AddParagraph();
                categoriesSummaryParagraph.AddLineBreak();
                categoriesSummaryParagraph.AddLineBreak();
                categoriesSummaryParagraph.AddFormattedText("Category Name: " + outputCategory.Name, StyleLibrary.TextFontBold);
                categoriesSummaryParagraph.AddLineBreak();

                foreach (FeatureDetail featureDetail in outputCategory.FeatureDetails)
                {
                    if (featureDetail.DisplayPreference == Constants.DisplayPreference.RESULT_PAGE || featureDetail.DisplayPreference == Constants.DisplayPreference.RESULT_AND_DETAIL_PAGE)
                    {
                        categoriesSummaryParagraph.AddLineBreak();
                        if (outputCategory.MOSValue.Equals(Properties.Resources.NA))
                        {
                            categoriesSummaryParagraph.AddFormattedText(featureDetail.ParameterName + ": " + Properties.Resources.NA , StyleLibrary.TextFont);
                        }
                        else
                        {
                            categoriesSummaryParagraph.AddFormattedText(featureDetail.ParameterName + ": " + featureDetail.Value.ToString("0.00") + " +/- " + featureDetail.StandardError.ToString("0.00"), StyleLibrary.TextFont);
                        }                        
                    }
                }
            }

            #endregion


            #region Photo Details Section

            foreach (Category category in result.outputCategoryList)
            {
                bool firstLine = true;

                foreach (Photo photo in category.PhotoList)
                {
                    // Category Details Section 
                    Section categoryDetailsSection = document.AddSection();

                    if (firstLine)
                    {
                        // Introductory paragraph
                        Paragraph categoryDetailsParagraphTitle = categoryDetailsSection.AddParagraph();
                        categoryDetailsParagraphTitle.AddFormattedText("Photos in  " + category.Name + " Category (" + category.PhotoList.Count + " photos)", StyleLibrary.HeadingFont);
                        
                        firstLine = false;
                    }

                    // Introductory paragraph
                    Paragraph paragraphTitle = categoryDetailsSection.AddParagraph();
                    paragraphTitle.AddLineBreak();
                    paragraphTitle.AddFormattedText("Details for " + photo.Filename, StyleLibrary.TextFontBold);
                    paragraphTitle.AddLineBreak();
                    paragraphTitle.AddLineBreak();

                    //Add the source photo
                    Image sourceImage = new Image(photo.SourceFilePath);
                    sourceImage.Height = 200;
                    sourceImage.Width = 400;

                    Paragraph sourceImageParagraph = categoryDetailsSection.AddParagraph();
                    sourceImageParagraph.Format.Alignment = ParagraphAlignment.Center;
                    sourceImageParagraph.Add(sourceImage);
                    sourceImageParagraph.AddLineBreak();
                    sourceImageParagraph.AddFormattedText("Source Photo", StyleLibrary.TextFont);
                    sourceImageParagraph.AddLineBreak();
                    sourceImageParagraph.AddLineBreak();

                    //Table with values for this photo
                    Table photoDetailsTable = categoryDetailsSection.AddTable();

                    Column photoParameterColumn = photoDetailsTable.AddColumn(Unit.FromCentimeter(6));
                    photoParameterColumn.Format.Alignment = ParagraphAlignment.Left;

                    Column photoValueColumn = photoDetailsTable.AddColumn(Unit.FromCentimeter(3));
                    photoValueColumn.Format.Alignment = ParagraphAlignment.Right;

                    //Add Header Row
                    Row photoHeaderRow = photoDetailsTable.AddRow();
                    photoHeaderRow.Cells[0].AddParagraph().AddFormattedText("Parameter", StyleLibrary.TextFontBold);
                    photoHeaderRow.Cells[1].AddParagraph().AddFormattedText("Value", StyleLibrary.TextFontBold);

                    //Add the details
                    foreach (PhotoDetail photoDetail in photo.PhotoDetails)
                    {
                        if (photoDetail.DisplayPreference == Constants.DisplayPreference.DETAIL_PAGE || photoDetail.DisplayPreference == Constants.DisplayPreference.RESULT_AND_DETAIL_PAGE)
                        {
                            Row detailRow = photoDetailsTable.AddRow();
                            detailRow.Cells[0].AddParagraph().AddFormattedText(photoDetail.ParameterName, StyleLibrary.TextFont);
                            detailRow.Cells[1].AddParagraph().AddFormattedText(photoDetail.Value.ToString("0.00"), StyleLibrary.TextFont);
                        }
                    }

                    //Add the visualization images 
                    Section visualizationSection = document.AddSection();
                    foreach (VisualizationImage visualizationImage in photo.VisualizationImages)
                    {
                        Image image = new Image(visualizationImage.FilePath);
                        image.Height = 200;
                        image.Width = 400;

                        Paragraph visualizationImageParagraph = visualizationSection.AddParagraph();
                        visualizationImageParagraph.AddLineBreak();
                        visualizationImageParagraph.Format.Alignment = ParagraphAlignment.Center;
                        visualizationImageParagraph.Add(image);
                        visualizationImageParagraph.AddLineBreak();
                        visualizationImageParagraph.AddFormattedText(visualizationImage.Visualization, StyleLibrary.TextFont);
                    }
                }
            }

            #endregion

            return document;
        }
    }
}
