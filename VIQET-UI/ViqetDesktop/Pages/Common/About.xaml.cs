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
    /// Interaction logic for About.xaml
    /// </summary>
    public partial class About : UserControl
    {
        public About()
        {
            InitializeComponent();
            this.Version.Text = Properties.Resources.Version + ": " + Properties.Resources.VersionName+ " " + Constants.Version;
        }

        private void OKButton_Click(object sender, RoutedEventArgs e)
        {
            this.Visibility = Visibility.Collapsed;
        }

        private void Navigate(object sender, RoutedEventArgs e)
        {
            System.Diagnostics.Process.Start("http://www.vqeg.org");
        }

        private void NavigateLicensePage(object sender, RoutedEventArgs e)
        {
            System.Diagnostics.Process.Start("http://www.eclipse.org/legal/epl-v10.html");
        }

    }
}
