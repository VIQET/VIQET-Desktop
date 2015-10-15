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
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace ViqetDesktop
{
    /// <summary>
    /// Interaction logic for ResultDetailPage.xaml
    /// </summary>
    public partial class ResultDetailPage : Page
    {
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

        Result result = null;
        ObservableCollection<CategoryListViewItem>  categoryListViewItems = null;
        CancellationTokenSource photoAnalyzerCancellationTokenSource = null;
        string deviceName = null;
        int count = 0;
        int photoCount = 0;

        public ResultDetailPage(Result result)
        {
            Initialize();

            this.result = result;
            this.ResultName.Text = this.result.Name;
            
            if (this.result.AllAnalyzed())
            {
                PopulateCategories(this.result.outputCategoryList);

                /* this.ProgressBarControl.Visibility = Visibility.Collapsed; */
                OverallProgressRing.Visibility = Visibility.Hidden;
            }
            else
            {
                this.photoCount = PopulateCategories(this.result.InputCategoryList);
                this.PhotoStatus.Text += photoCount;
                PrepareUIForProgressIndication(photoCount);
            }
        }

        private void Initialize()
        {
            InitializeComponent();
            this.categoryListViewItems = new ObservableCollection<CategoryListViewItem>();
            this.CategoryListView.DataContext = this.categoryListViewItems;
        }

        private void PrepareUIForProgressIndication(int photoCount)
        {
            //Make Progress Indicator visible
            count = 0;

            OverallProgressRing.Visibility = Visibility.Visible;
            OverallProgressRing.Start();
            this.PhotoStatus.Text = Properties.Resources.ProcessingPhoto + " 1 " + Properties.Resources.Of + " " + photoCount;
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

        public async Task<bool> Analyze(ObservableCollection<CategoryListViewItem> categoryListViewItems, String deviceName, CancellationToken cancellationToken)
        {
            //Delete the Temp Folder
            Directory.Delete(Constants.TempDirectory, true);

            PhotoInspector photoInspector = new PhotoInspector();

            for(int categoryIndex = 0; categoryIndex < categoryListViewItems.Count; categoryIndex++)
            {
                if (cancellationToken.IsCancellationRequested)
                    break;
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

                if (cancellationToken.IsCancellationRequested)
                    break;
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

            /*if (ProgressBarControl.Visibility == Visibility.Visible)
            {
                ProgressBarControl.Value += 1;
                this.PhotoStatus.Text = Properties.Resources.ProcessingPhoto + " " +(int)(ProgressBarControl.Value + 1) + " " + Properties.Resources.Of + " " + (int)ProgressBarControl.Maximum;
            }*/
            if(OverallProgressRing.Visibility == Visibility.Visible)
            {
                count++;
                this.PhotoStatus.Text = Properties.Resources.ProcessingPhoto + " " + (count+1) + " " + Properties.Resources.Of + " " + photoCount ;
            }
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
            this.VisualizationComboBox.SelectedIndex = 0; 

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
                    featureName.Style = this.Resources["ContentTextStyle"] as Style;
                    featureName.Foreground = new SolidColorBrush(Colors.Black);
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

            this.PhotoStatus.Text = Properties.Resources.Stopping;
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
