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
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace ViqetDesktop
{
    /// <summary>
    /// Interaction logic for ProgressRing.xaml
    /// </summary>
    public partial class ProgressRing : UserControl
    {
        public const string OFF = "off";
        public const string ON = "on";

        public static readonly DependencyProperty BallColor = DependencyProperty.Register("CircleBrush", typeof(Brush), typeof(ProgressRing), new FrameworkPropertyMetadata(new PropertyChangedCallback(AdjustControl)));
        public static readonly DependencyProperty Ring = DependencyProperty.Register("RingSpinning", typeof(string), typeof(ProgressRing), new FrameworkPropertyMetadata(new PropertyChangedCallback(ChangeSpinning)));
        private Storyboard myStoryboard;

        public Brush CircleBrush
        {
            get
            {
                return (Brush) GetValue(BallColor) ;//as SolidColorBrush;
            }
            set
            {
                SetValue(BallColor, value);
                UpdateColor();                
            }
        }

        private static void AdjustControl(DependencyObject source, DependencyPropertyChangedEventArgs e) 
        {
            (source as ProgressRing).UpdateColor();
        }

        public string RingSpinning
        {
            get
            {
                return (string)GetValue(Ring);
            }
            set
            {
                SetValue(Ring, value);
                UpdateSpinning(value);
            }
        }

        private static void ChangeSpinning(DependencyObject source, DependencyPropertyChangedEventArgs e)
        {
            (source as ProgressRing).UpdateSpinning((string)e.NewValue);
        }

        private void UpdateSpinning(string state)
        {
            if (state == ON)
            {
                this.Visibility = Visibility.Visible;
                this.Start();
            }
            else if(state == OFF)
            {
                this.Visibility = Visibility.Collapsed;
                this.Stop();
            }
        }

        private void UpdateColor()
        {
            int childCount = VisualTreeHelper.GetChildrenCount(this.ProgressRingGrid);
            VisualTreeHelper.GetChild(this.ProgressRingGrid, 0);
            for(int childIndex = 0; childIndex < childCount; childIndex++)
            {
                DependencyObject obj = VisualTreeHelper.GetChild(this.ProgressRingGrid, childIndex);
                if (obj.GetType() == typeof(Ellipse))
                {
                    Ellipse ellipse = obj as Ellipse;
                    ellipse.Fill = this.CircleBrush;
                }
            }
        }

        public ProgressRing()
        {
            InitializeComponent();
            
            RotateTransform rotateTransform = new RotateTransform(0);
            ProgressRingGrid.RenderTransform = rotateTransform;

            DoubleAnimation rotateAnimation = new DoubleAnimation();
            rotateAnimation.From = 0;
            rotateAnimation.To = 360;
            rotateAnimation.Duration = new Duration(TimeSpan.FromSeconds(1));
            rotateAnimation.RepeatBehavior = RepeatBehavior.Forever;

            myStoryboard = new Storyboard();
            myStoryboard.Children.Add(rotateAnimation);
            Storyboard.SetTarget(rotateAnimation, this.ProgressRingGrid);
            Storyboard.SetTargetProperty(rotateAnimation, new PropertyPath("(UIElement.RenderTransform).(RotateTransform.Angle)"));

            //Set Default Color
            SolidColorBrush solidColorBrush = new SolidColorBrush();
            solidColorBrush.Color = Color.FromArgb(255, 233, 234, 234);
            this.CircleBrush = solidColorBrush;
        }

        public void Start()
        {
            this.myStoryboard.Begin();
        }

        public void Stop()
        {
            this.myStoryboard.Stop();
        }


    }
}
