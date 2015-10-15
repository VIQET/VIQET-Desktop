/*
Copyright © 2015 Intel Corporation
This program and the accompanying materials are made available under the terms of the Eclipse Public License v1.0, 
 which accompanies this distribution, and is available at http://www.eclipse.org/legal/epl-v10.html . https://github.com/viqet
* Contributors:
*    Intel Corporation - initial API and implementation and/or initial documentation
*/
using System.Windows;
using System.Windows.Controls;

namespace ViqetDesktop
{
    /// <summary>
    /// Interaction logic for SettingsPage.xaml
    /// </summary>
    public partial class Setting : UserControl
    {
        public Setting()
        {  
            InitializeComponent();
            LoadSettings();
        }

        private void OKButton_Click(object sender, RoutedEventArgs e)
        {
            SaveSettings();
            GoBack();
        }

        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            GoBack();
        }

        private void GoBack()
        {
            LoadSettings();
            this.Visibility = Visibility.Collapsed;
        }

        private void SaveSettings()
        {
            Properties.Settings.Default.CheckPhotoCount = (bool)this.CheckPhotoCount.IsChecked;
            Properties.Settings.Default.DisplayPhotoMOS = (bool)this.DisplayPhotoMOS.IsChecked;
            Properties.Settings.Default.Save();
        }

        private void LoadSettings()
        {
            this.CheckPhotoCount.IsChecked = Properties.Settings.Default.CheckPhotoCount;
            this.DisplayPhotoMOS.IsChecked = Properties.Settings.Default.DisplayPhotoMOS;
        }
    }
}