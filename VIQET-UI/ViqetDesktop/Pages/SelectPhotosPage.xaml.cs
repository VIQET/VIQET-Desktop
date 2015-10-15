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
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;

namespace ViqetDesktop
{
    /// <summary>
    /// Interaction logic for SelectPhotosPage.xaml
    /// </summary>
    public partial class SelectPhotosPage : Page
    {
        //Item used to populate the listview
        private class PhotoItem
        {
            public ImageSource Thumbnail { get; set; }
            public string FilePath { get; set; }
        }
        
        Result result = null;
        private string placeHolderImagePath = "/Assets/Icons/add_photo.png";
        public SelectPhotosPage(Result result)
        {
            InitializeComponent();
            this.result = result;
            this.SelectPhotosGrid.DataContext = this.result;
            PlaceHolder();
        }

        public SelectPhotosPage()
        {
            InitializeComponent();
            this.result = new Result();
            this.SelectPhotosGrid.DataContext = this.result;
            AddAllPlaceHolders();
        }
        
        private void PlaceHolder()
        {
            foreach (string inputCategoryName in MethodologyProvider.Get().Methodology.InputCategories.Keys)
            {
                int count = GetCount(inputCategoryName);
                for (int i = 0; i < (5-count); i++)
                {
                    AddPlaceHolder(inputCategoryName);
                }
            }     
            this.SelectPhotosGrid.DataContext = this.result;
            UpdateDataContext();
        }

        private void AddPhotos_Click(object sender, RoutedEventArgs e)
        {
            Button buttonClicked = sender as Button;

            StackPanel parentStackPanel = buttonClicked.Parent as StackPanel;
            Grid parentGrid = parentStackPanel.Parent as Grid;
            TextBlock categoryName = parentGrid.FindName("CategoryName") as TextBlock;
            string inputCategoryName = categoryName.Text;

            //Display Open File Dialog
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Title = "Add a photo";
            openFileDialog.Filter = "Picture Files (*.jpg, *.bmp, *.png) |*.jpg;*.bmp;*.png |All Files (*.*)|*.*";
            openFileDialog.Multiselect = true;
            Nullable<bool> fileOpenResult = openFileDialog.ShowDialog();

            //Add Files to PhotoFetcher
            if (fileOpenResult == true)
            {
                string[] filenames = openFileDialog.FileNames;

                RemovePlaceHolders(inputCategoryName);

                for (int fileIndex = 0; fileIndex < filenames.Length; fileIndex++)
                {
                    Photo photo = new Photo();
                    photo.Filename = filenames.ElementAt(fileIndex);
                    photo.InputCategory = inputCategoryName;
                    photo.SourceFilePath = filenames[fileIndex];
                    photo.Filename = Path.GetFileName(filenames[fileIndex]);
                    if (photo.Filename.Equals("add_photo.png"))
                    {
                        photo.Name = "";
                    }
                    else
                    {
                        photo.Name = photo.Filename;
                    }
                    photo.OutputCategories = MethodologyProvider.Get().Methodology.InputCategories[inputCategoryName].outputCategories;

                    this.result.PhotoList.Add(photo);
                }
                AddAllPlaceHolders();
                UpdateDataContext();
            }
        }
        private int GetCount(string inputCategory)
        {
            int count = 0;
            for (int i = 0; i < this.result.PhotoList.Count; i++)
            {
                if (this.result.PhotoList[i].InputCategory.Equals(inputCategory))
                {
                    count++;
                }
            }
            return count;
        }

        private void RemovePlaceHolders(string inputCategoryName)
        {
            List<Photo> listPlaceHolder = new List<Photo>();
            for (int i = 0; i < result.PhotoList.Count; i++)
            {
                if (this.result.PhotoList[i].SourceFilePath.Equals(placeHolderImagePath) && this.result.PhotoList[i].InputCategory.Equals(inputCategoryName))
                {
                    Photo ph = this.result.PhotoList[i];
                    listPlaceHolder.Add(ph);
                }
            }
            foreach (Photo ph in listPlaceHolder)
            {
                this.result.PhotoList.Remove(ph);
            }
        }

        private void RemoveAllPlaceHolders()
        {
            List<Photo> listPlaceHolder = new List<Photo>();
            for (int i = 0; i < result.PhotoList.Count; i++)
            {
                if (this.result.PhotoList[i].SourceFilePath.Equals(placeHolderImagePath))
                {
                    Photo ph = this.result.PhotoList[i];
                    listPlaceHolder.Add(ph);
                }
            }
            foreach (Photo ph in listPlaceHolder)
            {
                this.result.PhotoList.Remove(ph);
            }
        }

        private void AddPlaceHolder(string inputCategoryName)
        {
            Photo photo = new Photo();
            photo.InputCategory = inputCategoryName;
            photo.SourceFilePath = placeHolderImagePath;
            photo.Filename = Path.GetFileName(placeHolderImagePath);
            if (photo.Filename.Equals("add_photo.png"))
            {
                photo.Name = "";
            }
            else
            {
                photo.Name = photo.Filename;
            }
            photo.OutputCategories = MethodologyProvider.Get().Methodology.InputCategories[inputCategoryName].outputCategories;//Get the list of outputCategories from the MethodologyProvider
            this.result.PhotoList.Add(photo);
        }

        private void AddGroupPlaceHolders(string inputCategoryName)
        {
            int totalPlaceholders = GetCount(inputCategoryName);
            while (totalPlaceholders < 5)
            {
                AddPlaceHolder(inputCategoryName);
                totalPlaceholders++;
            }
        }
        private void AddAllPlaceHolders()
        {
            //Fetch input categories from Methodology
            Methodology methodology = MethodologyProvider.Get().Methodology;

            //Check if any category doesnt have the required number of photos
            foreach (Category category in this.result.InputCategoryList)
            {
                AddGroupPlaceHolders(category.Name);
            }
        }

        private async void DeleteAllPhotos_Click(object sender, RoutedEventArgs e)
        {
            if (MessageBox.Show(Properties.Resources.DeletePhotosConfirmationMessage, Properties.Resources.DeleteMultipleItems, MessageBoxButton.OKCancel) == MessageBoxResult.OK)
            {
                Button buttonClicked = sender as Button;

                StackPanel parentStackPanel = buttonClicked.Parent as StackPanel;
                Grid parentGrid = parentStackPanel.Parent as Grid;
                TextBlock categoryName = parentGrid.FindName("CategoryName") as TextBlock;
                ListView categoryListView = parentGrid.FindName("PhotoListView") as ListView;

                string inputCategoryName = categoryName.Text;

                //Create list of photos to delete
                List<Photo> photosToDelete = new List<Photo>();
                foreach (Photo photo in categoryListView.Items)
                {
                    if (!(photo.SourceFilePath.Equals(placeHolderImagePath)))
                    {
                        photosToDelete.Add(photo);
                    }
                }

                //Delete the photos
                foreach (Photo photo in photosToDelete)
                {
                    this.result.PhotoList.Remove(photo);
                }
                AddAllPlaceHolders();
                UpdateDataContext();
                await Task.Delay(50);
            }
        }

        private async void DeletePhotos_Click(object sender, RoutedEventArgs e)
        {
            Button buttonClicked = sender as Button;

            StackPanel parentStackPanel = buttonClicked.Parent as StackPanel;
            Grid parentGrid = parentStackPanel.Parent as Grid;
            TextBlock categoryName = parentGrid.FindName("CategoryName") as TextBlock;
            ListView categoryListView = parentGrid.FindName("PhotoListView") as ListView;

            string inputCategoryName = categoryName.Text;

            //Create list of photos to delete
            List<Photo> photosToDelete = new List<Photo>();
            foreach (Photo photo in categoryListView.SelectedItems)
            {
                if (!(photo.SourceFilePath.Equals(placeHolderImagePath)))
                {
                    photosToDelete.Add(photo);
                }
            }

            //Delete the photos
            foreach (Photo photo in photosToDelete)
            {
                this.result.PhotoList.Remove(photo);
            }
            AddAllPlaceHolders();
            UpdateDataContext();
            await Task.Delay(50);
        }


        private void UpdateDataContext()
        {
            //Update the binding (Done manually since we are not using an observable collection)
            this.SelectPhotosGrid.DataContext = null;
            this.SelectPhotosGrid.DataContext = this.result;
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

        Boolean NoPhotos = false;
        private bool IsMethodologyFollowed()
        {
            //Check if there are no photos
            if (this.result.PhotoList.Count <= 0)
            {
                NoPhotos = true;
                return false;
            }

            //Check if methodology is followed
            if (Properties.Settings.Default.CheckPhotoCount)
            {
                //Fetch input categories from Methodology
                Methodology methodology = MethodologyProvider.Get().Methodology;
                
                //Check if any category doesnt have the required number of photos
                foreach (Category category in this.result.InputCategoryList)
                {
                    if (category.PhotoList.Count < methodology.InputCategories[category.Name].RequiredPhotoCount)
                    {
                        return false;
                    }
                }
            }
                
            //If none of the checks fail, return true
            return true;
        }

        private void AnalyzeButton_Click(object sender, RoutedEventArgs e)
        {
            NoPhotos = false;
            if (this.DeviceName.Text.Equals("")|| this.DeviceName.Text.Equals("Enter Test Name Here"))
            {
                MessageBox.Show(Properties.Resources.IncorrectName, Properties.Resources.Error, MessageBoxButton.OK, MessageBoxImage.Warning);
            }
            else
            {
                RemoveAllPlaceHolders();
                this.result.Name = this.DeviceName.Text;
                if (IsMethodologyFollowed())
                {
                    NavigationService navigationService = NavigationService.GetNavigationService(this);
                    if (this.result.AllAnalyzed())
                    {
                        //Either no change made to result, or a photo was deleted. Save the result and go directly to the Results Page
                        ResultStore.SaveResult(this.result);
                        navigationService.Navigate(new ResultPage(this.result, 1));
                    }
                    else
                    {
                        //Some photos havent been analyzed, so go to ResultDetailPage
                        navigationService.Navigate(new ResultDetailPage(this.result));
                    }
                }
                else
                {
                    AddAllPlaceHolders();
                    MessageBox.Show(Properties.Resources.NoPhotosSelected, Properties.Resources.Error,MessageBoxButton.OK ,MessageBoxImage.Warning);
                }
            }            
        }

        private void BackButton_Click(object sender, RoutedEventArgs e)
        {
            if (MessageBox.Show(Properties.Resources.ExitSelectPhotosWarning, Properties.Resources.BackToMainScreenWarning, MessageBoxButton.OKCancel) == MessageBoxResult.OK)
            {
                NavigationService navigationService = NavigationService.GetNavigationService(this);
                navigationService.GoBack();
            }
        }

        private void Help_Click(object sender, RoutedEventArgs e)
        {
            this.HelpOverlay.Visibility = Visibility.Visible;
        }

        private void AboutButton_Click(object sender, RoutedEventArgs e)
      {
            this.AboutOverlay.Visibility = Visibility.Visible;
       }

        private void DeviceName_GotFocus(object sender, RoutedEventArgs e)
        {
            TextBox deviceName = sender as TextBox;
            deviceName.Clear();
        }

        private void InputCategoryInfo_Click(object sender, RoutedEventArgs e)
        {
            Button buttonClicked = sender as Button;

            StackPanel parentStackPanel = buttonClicked.Parent as StackPanel;
            Grid parentGrid = parentStackPanel.Parent as Grid;
            TextBlock categoryName = parentGrid.FindName("CategoryName") as TextBlock;
            string inputCategoryName = categoryName.Text;
            InputCategoryInfo inputCategoryInfo = MethodologyProvider.Get().Methodology.InputCategories[inputCategoryName];
            
            this.InstructionOverlay.Scroll.ScrollToTop();
            this.InstructionOverlay.CategoryInfoGrid.DataContext = inputCategoryInfo;
            this.InstructionOverlay.GeneralInfoGrid.DataContext = MethodologyProvider.Get().Methodology;
            this.InstructionOverlay.Visibility = Visibility.Visible;
        }
    }
}
