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
using System.Linq;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;

namespace ViqetDesktop
{
    class StarImage
    {
        public string SourceFilePath { get; set; }
        public BitmapImage Thumbnail
        {
            get { return CommonTasks.FetchImage(this.SourceFilePath, 100); }
        }
        public StarImage(string filePath)
        {
            this.SourceFilePath = filePath;
        }
    }

    class ResultTableItem1
    {
        public String ParameterName { get; set; }
        public List <String> Values { get; set; }
    }

    class ResultPageMOS
    {
        public String Name { get; set; }
        public String Value = Properties.Resources.NA;
        public List<BitmapImage> listOfStars { get; set; }
    }

    class ResultTable
    {
        private List<Category> outputCategoryList;
        public ResultTable(List<Category> outputCategoryList)
        {
            this.outputCategoryList = outputCategoryList;
        }
    }
    
    /// <summary>
    /// Interaction logic for ResultPage.xaml
    /// </summary>
    public partial class ResultPage : Page
    { 
        Result result = null;
        ResultTable resultTable = null;
        int backCount = 0;

        public ResultPage(Result result, int backCount)
        {
            InitializeComponent();
            this.result = result;
            this.resultTable = new ResultTable(result.outputCategoryList);
            this.backCount = backCount;
            this.DeviceName.Text = result.Name;

            //MOS
            FeatureDetail mosDetail = this.result.GetOverallMOS();
            double mosDetailValue = Math.Round(mosDetail.Value, 1);
            this.MOS.Text = double.IsNaN(mosDetail.Value)? Properties.Resources.NA : mosDetailValue.ToString("0.0");
            Grid[] stars = new Grid[5] { this.Star1, this.Star2, this.Star3, this.Star4, this.Star5 };
            PopulateStars(mosDetailValue, stars);
            this.ConfidenceInterval.Text = "Variability: +/- " + mosDetail.StandardError.ToString("0.0");

            CheckErrors();

            //Category MOS
            this.CategoryMOSList.DataContext = this.result.outputCategoryList;
            this.OverallDetailsListHeader.DataContext = this.result.outputCategoryList;
            this.OverallDetailsList.DataContext = GetResultTableItems(this.result);

            ResultDetailPageInit();
        }

        public void CheckErrors()
        {
            if (this.result.MethodologyFollowed)
            {
                if (this.result.GetOverallMOS().Value.ToString().Equals("NaN"))
                {
                    this.FlatRegionError.Visibility = Visibility.Visible;
                    this.FlatRegionError.Text = Properties.Resources.FlatRegionError + Properties.Resources.NoteAboutMinPhotos;
                }
                else
                {
                    this.FlatRegionError.Visibility = Visibility.Visible;
                    this.FlatRegionError.Text = Properties.Resources.FlatRegionError;
                }
            }
            else
            {
                if (this.result.GetOverallMOS().Value.ToString().Equals("NaN"))
                {
                    this.FlatRegionError.Visibility = Visibility.Visible;
                    this.FlatRegionError.Text = Properties.Resources.NoteAboutMinPhotos;
                }
                else
                {
                    this.FlatRegionError.Visibility = Visibility.Collapsed;
                }
            }
            this.FlatRegionError.Foreground = Brushes.Black;

            foreach (Category category in result.outputCategoryList)
            {
                if (category.MOSValue.Equals(Properties.Resources.NA))
                {
                    this.CategoryMOSErrorTextBox.Visibility = Visibility.Visible;
                    this.CategoryMOSErrorTextBox.Text = Properties.Resources.NoteAboutMinPhotosCategory;
                    break;
                }
                else
                {
                    this.CategoryMOSErrorTextBox.Visibility = Visibility.Collapsed;
                }
            }
            this.CategoryMOSErrorTextBox.Foreground = Brushes.Black;
        }

        private List<ResultTableItem1> GetResultTableItems(Result result)
        {
            Dictionary<String,ResultTableItem1> resultTableItems = new Dictionary<String,ResultTableItem1>();


            foreach (Category category in result.outputCategoryList)
            {
                foreach (FeatureDetail featureDetail in category.FeatureDetails)
                {
                    if (featureDetail.DisplayPreference == Constants.DisplayPreference.RESULT_PAGE || featureDetail.DisplayPreference == Constants.DisplayPreference.RESULT_AND_DETAIL_PAGE)
                    {
                        if (!resultTableItems.ContainsKey(featureDetail.ParameterName))
                        {
                            ResultTableItem1 resultTableItem = new ResultTableItem1();
                            resultTableItem.ParameterName = featureDetail.ParameterName;
                            resultTableItem.Values = new List<string>();
                            resultTableItems.Add(featureDetail.ParameterName, resultTableItem);
                        }
                        if (category.MOSValue.Equals(Properties.Resources.NA))
                        {
                            resultTableItems[featureDetail.ParameterName].Values.Add(Properties.Resources.NA);
                        }
                        else
                        {
                            resultTableItems[featureDetail.ParameterName].Values.Add(featureDetail.Value.ToString("0.00") + " +/- " + featureDetail.StandardError.ToString("0.00"));
                        }
                    }
                }
            }
            return resultTableItems.Values.ToList<ResultTableItem1>();
        }

        //# region UI Buttons

        private void ChangeGridBackground(Grid grid, string imageName)
        {
            ImageSource imageSource = new BitmapImage(new Uri("pack://application:,,,/ViqetDesktop;component" + imageName));
            ImageBrush imageBrush = new ImageBrush(imageSource);
            grid.Background = imageBrush;
        }

        private void PopulateStars(double mos, Grid[] stars)
        {
            for (int starIndex = 0; starIndex < stars.Count(); starIndex++)
            {
                double starValue = mos - starIndex;

                if (starValue > 0.9) ChangeGridBackground(stars[starIndex], "/Assets/Icons/Star1.0.png");
                else if (starValue > 0.8) ChangeGridBackground(stars[starIndex], "/Assets/Icons/Star0.9.png");
                else if (starValue > 0.7) ChangeGridBackground(stars[starIndex], "/Assets/Icons/Star0.8.png");
                else if (starValue > 0.6) ChangeGridBackground(stars[starIndex], "/Assets/Icons/Star0.7.png");
                else if (starValue > 0.5) ChangeGridBackground(stars[starIndex], "/Assets/Icons/Star0.6.png");
                else if (starValue > 0.4) ChangeGridBackground(stars[starIndex], "/Assets/Icons/Star0.5.png");
                else if (starValue > 0.3) ChangeGridBackground(stars[starIndex], "/Assets/Icons/Star0.4.png");
                else if (starValue > 0.2) ChangeGridBackground(stars[starIndex], "/Assets/Icons/Star0.3.png");
                else if (starValue > 0.1) ChangeGridBackground(stars[starIndex], "/Assets/Icons/Star0.2.png");
                else if (starValue > 0.0) ChangeGridBackground(stars[starIndex], "/Assets/Icons/Star0.1.png");
                else ChangeGridBackground(stars[starIndex], "/Assets/Icons/Star0.0.png");   
            }
        }

        private void BackButton_Click(object sender, RoutedEventArgs e)
        {
            NavigationService navigationService = NavigationService.GetNavigationService(this);
            while (backCount > 1)
            {
                navigationService.RemoveBackEntry();
                backCount--;
            }
            navigationService.GoBack();
        }

        private void FinishedButton_Click(object sender, RoutedEventArgs e)
        {
            NavigationService navigationService = NavigationService.GetNavigationService(this);
            while (navigationService.CanGoBack)
            {
                navigationService.GoBack();
            }   
        }

        private void Help_Click(object sender, RoutedEventArgs e)
        {
            this.HelpOverlay.Visibility = Visibility.Visible;
        }

        private void ExportXMLButton_Click(object sender, RoutedEventArgs e)
        {
            ResultsExporter.ExportToXML(this.result);
        }

        private void ExportCSVButton_Click(object sender, RoutedEventArgs e)
        {
            ResultsExporter.ExportToCSV(this.result);
        }

        private void ExportPDFButton_Click(object sender, RoutedEventArgs e)
        {
            ResultsExporter.ExportToPDF(this.result);
        }

        private void DetailedResultsButton_Click(object sender, RoutedEventArgs e)
        {
            NavigationService navigationService = NavigationService.GetNavigationService(this);
            NavigationService.Navigate(new ResultDetailPage(this.result));
        }

        private void AboutButton_Click(object sender, RoutedEventArgs e)
        {
            this.AboutOverlay.Visibility = Visibility.Visible;
        }

        public class CategoryListViewItem
                {
                    public string Name { get; set; }
                    public bool IsExpanded { get; set; }
                    public ObservableCollection<PhotoListViewItem> PhotoListViewItems { get; set; }

                    public CategoryListViewItem()
                    {
                        PhotoListViewItems = new ObservableCollection<PhotoListViewItem>();
                        IsExpanded = false;
                    }
                }
        
        public class PhotoListViewItem
                {
                    public Photo Photo { get; set; }
                    public string Filename { get; set; }
                    public ImageSource Thumbnail { get; set; }
                    public bool Enabled { get; set; }
                    public string Spinning { get; set; }

                    public PhotoListViewItem()
                    {
                        this.Spinning = ProgressRing.OFF;
                        this.Enabled = true;
                    }
                }
        
        ObservableCollection<CategoryListViewItem>  categoryListViewItems = null;
        CancellationTokenSource photoAnalyzerCancellationTokenSource = null;
        string deviceName = null; 

        public void ResultDetailPageInit()
                {
                    Initialize();
            
                    if (this.result.AllAnalyzed())
                    {
                        PopulateCategories(this.result.outputCategoryList);
                    }
                    else
                    {
                        int photoCount = PopulateCategories(this.result.InputCategoryList);
                    }
                }

        private int PopulateCategories(List<Category> categoryList)
                {
                    PhotoListViewItem firstPhotoListViewItem = null;

                    int photoCount = 0;
                    if (categoryList.Count > 0)
                    {
                        //Put the different categories into different listViews
                        foreach (Category category in categoryList)
                        {
                            CategoryListViewItem categoryListViewItem = new CategoryListViewItem();
                            categoryListViewItem.Name = category.Name;
                            this.categoryListViewItems.Add(categoryListViewItem);

                            if (category.PhotoList.Count > 0)
                            {
                                foreach (Photo photo in category.PhotoList)
                                {
                                    PhotoListViewItem photoListViewItem = CreatePhotoListViewItem(photo, true);
                                    categoryListViewItem.PhotoListViewItems.Add(photoListViewItem);
                            
                                    if (!photo.Analyzed)
                                    {
                                        photoListViewItem.Enabled = false;
                                    }
                            
                                    //Save the first photo item to display
                                    if (firstPhotoListViewItem == null)
                                    {
                                        firstPhotoListViewItem = photoListViewItem;
                                    }
                                }

                                photoCount += category.PhotoList.Count;
                            }
                        }

                        //Display the details of the first Photo
                        PopulatePhotoDetails(firstPhotoListViewItem);

                    }
                    return photoCount;
                }

        private PhotoListViewItem CreatePhotoListViewItem(Photo photo, bool enabled)
                {
                    PhotoListViewItem photoListViewItem = new PhotoListViewItem();
                    photoListViewItem.Photo = photo;
                    photoListViewItem.Filename = photo.Filename;
                    photoListViewItem.Thumbnail = FetchImage(photo.SourceFilePath, 100);
                    photoListViewItem.Enabled = enabled;
                    photoListViewItem.Spinning = ProgressRing.OFF;
                    return photoListViewItem;
                }

        private void Initialize()
                {
                    InitializeComponent();
                    this.categoryListViewItems = new ObservableCollection<CategoryListViewItem>();
                    this.CategoryListView.DataContext = this.categoryListViewItems;
                    ExportComboBox.Items.Add("PDF");
                    ExportComboBox.Items.Add("CSV");
                    ExportComboBox.Items.Add("XML");
                    ExportComboBox.SelectionChanged += ExportComboBox_SelectionChanged;
            
                }

        private void ExportComboBox_SelectionChanged(object sender, RoutedEventArgs e)
                {
                    if (ExportComboBox.SelectedIndex == 0)
                        ResultsExporter.ExportToPDF(this.result);
                    else if (ExportComboBox.SelectedIndex == 1)
                        ResultsExporter.ExportToCSV(this.result);
                    else if (ExportComboBox.SelectedIndex == 2)
                        ResultsExporter.ExportToXML(this.result);

                    ExportComboBox.SelectedIndex = -1;
                }

        public async Task<bool> Analyze(ObservableCollection<CategoryListViewItem> categoryListViewItems, String deviceName, CancellationToken cancellationToken)
                {
                    //Delete the Temp Folder
                    Directory.Delete(Constants.TempDirectory, true);

                    PhotoInspector photoInspector = new PhotoInspector();

                    for(int categoryIndex = 0; categoryIndex < categoryListViewItems.Count; categoryIndex++)
                    {
                        Category category = new Category();
                        category.Name = categoryListViewItems[categoryIndex].Name;

                        //Expand the category
                        CategoryListViewItem newItem = new CategoryListViewItem();
                        newItem.Name = categoryListViewItems[categoryIndex].Name;
                        newItem.PhotoListViewItems = categoryListViewItems[categoryIndex].PhotoListViewItems;
                        newItem.IsExpanded = true;
                        categoryListViewItems[categoryIndex] = newItem;

                        //Using for loop instead of foreach because we edit the listview
                        for (int index = 0; index < categoryListViewItems[categoryIndex].PhotoListViewItems.Count; index++)
                        {
                            MarkPhotoAsBusy(categoryListViewItems[categoryIndex], index);

                            Task analysisTask = new Task(() =>
                            {
                                Photo photo = categoryListViewItems[categoryIndex].PhotoListViewItems[index].Photo;
                                string categoryName = categoryListViewItems[categoryIndex].Name;

                                //Check if the photo has been analyzed before
                                if (photo.PhotoDetails.Count == 0)
                                {
                                    photoInspector.InspectAndUpdatePhoto(photo, categoryName);
                                }
                            });

                            analysisTask.Start();
                            await analysisTask;

                            MarkPhotoAsCompleted(categoryListViewItems[categoryIndex], index);

                            if (cancellationToken.IsCancellationRequested)
                                break;
                        }

                        //Collapse the category
                        newItem = new CategoryListViewItem();
                        newItem.Name = categoryListViewItems[categoryIndex].Name;
                        newItem.PhotoListViewItems = categoryListViewItems[categoryIndex].PhotoListViewItems;
                        newItem.IsExpanded = false;
                        categoryListViewItems[categoryIndex] = newItem;
                    }

                    if (!cancellationToken.IsCancellationRequested)
                    {
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }

        public void MarkPhotoAsBusy(CategoryListViewItem categoryListViewItem, int photoIndex)
                {
                    //Create a new PhotoListView Item and replace the old one (Since updating newItem.Spinning isnt reflected in the ListView)
                    PhotoListViewItem oldItem = categoryListViewItem.PhotoListViewItems.ElementAt(photoIndex);
                    PhotoListViewItem newItem = new PhotoListViewItem();
                    newItem.Photo = oldItem.Photo;
                    newItem.Filename = oldItem.Filename;
                    newItem.Enabled = false;
                    newItem.Thumbnail = oldItem.Thumbnail;
                    newItem.Spinning = ProgressRing.ON;

                    categoryListViewItem.PhotoListViewItems[photoIndex] = newItem;
                }

        public void MarkPhotoAsCompleted(CategoryListViewItem categoryListViewItem, int photoIndex)
                {
                    //Create a new PhotoListView Item and replace the old one (Since updating newItem.Spinning isnt reflected in the ListView)
                    PhotoListViewItem oldItem = categoryListViewItem.PhotoListViewItems.ElementAt(photoIndex);
                    PhotoListViewItem newItem = new PhotoListViewItem();
                    newItem.Photo = oldItem.Photo;
                    newItem.Filename = oldItem.Filename;
                    newItem.Enabled = true;
                    newItem.Thumbnail = oldItem.Thumbnail;
                    newItem.Spinning = ProgressRing.OFF;

                    categoryListViewItem.PhotoListViewItems[photoIndex] = newItem;

                    PopulatePhotoDetails(newItem);
                }

        private BitmapImage FetchImage(string fileName, int decodePixelWidth)
                {
                    BitmapImage bitmapImage = new BitmapImage();
                    bitmapImage.BeginInit();
                    bitmapImage.DecodePixelWidth = decodePixelWidth;
                    bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                    bitmapImage.UriSource = new Uri(fileName);
                    bitmapImage.EndInit();
                    return bitmapImage;
                }

        private void PopulatePhotoDetails(PhotoListViewItem photoListViewItem)
                {
                    //Set Photo Name
                    this.FileName.Text = photoListViewItem.Filename;

                    //Set Source Photo
                    this.SourcePhoto.Source = FetchImage(photoListViewItem.Photo.SourceFilePath, 500);

                    //Visualizations
                    this.VisualizationComboBox.DataContext = photoListViewItem.Photo.VisualizationImages;
                    this.VisualizationComboBox.SelectedIndex = 0; //TODO: Check if this causes a crash if it is run before the combo box is populated

                    //Set Photo Details
                    this.PhotoDetails.Children.Clear();
            
                    //double itemWidth = this.PhotoDetails.Width / 4;
                    Thickness itemMargin = new Thickness(20, 0, 20, 0);

                    foreach (PhotoDetail photoDetail in photoListViewItem.Photo.PhotoDetails)
                    {
                        if ((photoDetail.DisplayPreference == Constants.DisplayPreference.DETAIL_PAGE || photoDetail.DisplayPreference == Constants.DisplayPreference.RESULT_AND_DETAIL_PAGE) || 
                            (Properties.Settings.Default.DisplayPhotoMOS && photoDetail.ParameterName == "MOS"))
                        {
                            //Create Grid with Feature Name and Value
                            TextBlock featureName = new TextBlock();
                            featureName.Style = this.Resources["ContentHeadingTextStyle"] as Style;
                            //featureName.Margin. Left = 10;
                            featureName.TextAlignment = TextAlignment.Left;
                            featureName.Text = photoDetail.ParameterName;

                            TextBlock featureValue = new TextBlock();
                            featureValue.Style = this.Resources["ContentTextStyle"] as Style;
                            featureValue.TextAlignment = TextAlignment.Right;
                            featureValue.Text = double.IsNaN(photoDetail.Value) ? photoDetail.ValueString : photoDetail.Value.ToString("0.00");

                            Grid grid = new Grid();
                            grid.Height = 30;
                            grid.Width = 300;
                            grid.Margin = itemMargin;

                            grid.Children.Add(featureName);
                            grid.Children.Add(featureValue);

                            this.PhotoDetails.Children.Add(grid);
                        }
                    }
                }

        private async void Page_Loaded(object sender, RoutedEventArgs e)
                {
                    if (!this.result.AllAnalyzed())
                    {
                        this.photoAnalyzerCancellationTokenSource = new CancellationTokenSource();
                
                        bool success = await Analyze(this.categoryListViewItems, this.deviceName, this.photoAnalyzerCancellationTokenSource.Token);

                        NavigationService navigationService = NavigationService.GetNavigationService(this);
                        if (success)
                        {
                            navigationService.Navigate(new ResultPage(this.result, 2));
                            ResultStore.SaveResult(this.result);
                        }
                        else
                        {
                            navigationService.GoBack();
                        }
                    }
                }

        private void StopButton_Click(object sender, RoutedEventArgs e)
                {
                    this.photoAnalyzerCancellationTokenSource.Cancel();

                    Button stopButton = sender as Button;
                    stopButton.IsEnabled = false;
                }

        private void VisualizationComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
                {
                    if (e.AddedItems.Count > 0)
                    {
                        VisualizationImage visualizationImage = e.AddedItems[0] as VisualizationImage;
                        this.VisualizationPhoto.Source = FetchImage(visualizationImage.FilePath, 500);
                    }  
                }

        private void PhotoListView_MouseUp(object sender, MouseButtonEventArgs e)
                {
                    ListView photoListView = sender as ListView;
                    PhotoListViewItem photoListViewItem = photoListView.SelectedItem as PhotoListViewItem;
                    if (photoListViewItem != null && photoListViewItem.Enabled == true)
                    {
                        PopulatePhotoDetails(photoListViewItem);
                    }
                }  
    }


}
