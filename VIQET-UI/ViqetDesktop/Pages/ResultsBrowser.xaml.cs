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
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Navigation;

namespace ViqetDesktop
{
    public class ResultBrowserPageItem
    {
        public string Name { get; set; }
        public int PhotoCount { get; set; }
        public string OverallMOS { get; set; }
        public ObservableCollection<string> CategoryMOSList { get; private set; }

        public ResultBrowserPageItem()
        {
            this.Name = string.Empty;
            this.OverallMOS = string.Empty;
            this.CategoryMOSList = new ObservableCollection<string>();
        }
    }
    
    /// <summary>
    /// Interaction logic for ResultsBrowser.xaml
    /// </summary>
    public partial class ResultsBrowser : Page
    {
        ObservableCollection<ResultBrowserPageItem> resultBrowserPageItems = null;

        public ResultsBrowser()
        {
            InitializeComponent();
            DisplayResults();
            UpdateButtonVisibility();

            ObservableCollection<string> resultsHeaderItems = new ObservableCollection<string>();
            resultsHeaderItems.Add("Outdoor Day");
            resultsHeaderItems.Add("Indoor");
            resultsHeaderItems.Add("Outdoor Night");
            this.ResultsHeaderListView.DataContext = resultsHeaderItems;

        }

        private void DisplayResults()
        {
            List<Result> resultList = ResultStore.GetResultList();
            
            this.resultBrowserPageItems = new ObservableCollection<ResultBrowserPageItem>();
            foreach (Result result in resultList)
            {
                ResultBrowserPageItem resultDetailPageItem = new ResultBrowserPageItem();
                
                //Get Name
                resultDetailPageItem.Name = result.Name;

                //Get Photo Count
                resultDetailPageItem.PhotoCount = result.PhotoCount();
                
                //Get MOS
                FeatureDetail overallMOSDetail = result.GetOverallMOS();
                resultDetailPageItem.OverallMOS = double.IsNaN(overallMOSDetail.Value)? Properties.Resources.NA : overallMOSDetail.Value.ToString("0.0");

                //Get Category MOS
                foreach (Category category in result.outputCategoryList)
                {
                    resultDetailPageItem.CategoryMOSList.Add(category.MOSValue);
                }

                resultBrowserPageItems.Add(resultDetailPageItem);
            }
            this.ResultsListView.DataContext = resultBrowserPageItems;

            //Update Header
            if (resultList!=null && resultList.Count > 0)
            {
                ObservableCollection<string> resultsHeaderItems = new ObservableCollection<string>();
                foreach (Category category in resultList[0].outputCategoryList)
                {
                    resultsHeaderItems.Add(category.Name);
                }
                this.ResultsHeaderListView.DataContext = resultsHeaderItems;
            }
        }

        private async void DeleteAllButton_Click(object sender, RoutedEventArgs e)
        {
            if (MessageBox.Show(Properties.Resources.DeleteTestsConfirmationMessage, Properties.Resources.DeleteMultipleItems, MessageBoxButton.OKCancel) == MessageBoxResult.OK)
            {
                while (this.ResultsListView.Items.Count > 0)
                {
                    this.resultBrowserPageItems.RemoveAt(0);
                    await Task.Delay(50);
                }
                ResultStore.DeleteAllResults();
                UpdateButtonVisibility();
            }
        }

        private async void DeleteSelectedButton_Click(object sender, RoutedEventArgs e)
        {
            while (this.ResultsListView.SelectedItems.Count > 0)
            {
                int selectedIndex = this.ResultsListView.SelectedIndex;

                List<Result> resultList = ResultStore.GetResultList();
                if (selectedIndex < resultList.Count)
                {
                    Result result = resultList.ElementAt(selectedIndex);
                    this.resultBrowserPageItems.RemoveAt(selectedIndex);
                    ResultStore.DeleteResult(result);
                    await Task.Delay(50);
                }
            }
            UpdateButtonVisibility();
        }

        private void OpenSelectedButton_Click(object sender, RoutedEventArgs e)
        {
            int selectedIndex = this.ResultsListView.SelectedIndex;

            List<Result> resultList = ResultStore.GetResultList();
            if (selectedIndex >= 0 && selectedIndex < resultList.Count)
            {
                Result result = resultList.ElementAt(selectedIndex);
                if (result != null && File.Exists(result.FilePath))
                {
                    NavigationService navigationService = NavigationService.GetNavigationService(this);
                    navigationService.Navigate(new ResultPage(result, 1));
                }
                else
                {
                    MessageBox.Show(Properties.Resources.ResultDoesntExistError, Properties.Resources.ResultDeletedError);
                    DisplayResults();
                }
            }
        }

        private void ResultsListView_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            OpenSelectedButton_Click(sender, e);
        }

        private void NewTestButton_Click(object sender, RoutedEventArgs e)
        {
            NavigationService navigationService = NavigationService.GetNavigationService(this);
            navigationService.Navigate(new SelectPhotosPage());
            UpdateButtonVisibility();
        }

        private void EditTestButton_Click(object sender, RoutedEventArgs e)
        {
            int selectedIndex = this.ResultsListView.SelectedIndex;
            List<Result> resultList = ResultStore.GetResultList();
            if (selectedIndex >= 0 && selectedIndex < resultList.Count)
            {
                Result result = resultList.ElementAt(selectedIndex);
                if (result != null && File.Exists(result.FilePath))
                {
                    NavigationService navigationService = NavigationService.GetNavigationService(this);
                    navigationService.Navigate(new SelectPhotosPage(result));
                }
                else
                {
                    MessageBox.Show(Properties.Resources.ResultDoesntExistError, Properties.Resources.ResultDeletedError);
                    DisplayResults();
                }
            }
        }

        private void AboutButton_Click(object sender, RoutedEventArgs e)
        {
            this.AboutOverlay.Visibility = Visibility.Visible;
        }

        private void UpdateButtonVisibility()
        {
            if (this.resultBrowserPageItems == null || this.resultBrowserPageItems.Count == 0)
            {
                this.TestButtons.Visibility = Visibility.Collapsed;
            }
            else
            {
                this.TestButtons.Visibility = Visibility.Visible;
            }
        }

        private void DismissOverlay_Click(object sender, RoutedEventArgs e)
        {
            Properties.Settings.Default.FirstRun = false;
            Properties.Settings.Default.Save();
        }

        private void Help_Click(object sender, RoutedEventArgs e)
        {
            this.HelpOverlay.Visibility = Visibility.Visible;
        }

        private void Page_Loaded(object sender, RoutedEventArgs e)
        {
        }
    }
}
