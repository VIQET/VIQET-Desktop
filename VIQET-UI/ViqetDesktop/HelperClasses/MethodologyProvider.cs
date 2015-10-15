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
using System.Runtime.Serialization.Json;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace ViqetDesktop
{
    public class MethodologyProvider
    {
        public Methodology Methodology {get; private set;}

        private static MethodologyProvider methodologyProvider = null;
        private MethodologyProvider()
        {
            this.Methodology = FetchMethodologyFromFile();
        }

        public static MethodologyProvider Get()
        {
            if (methodologyProvider == null)
            {
                methodologyProvider = new MethodologyProvider();
            }
            return methodologyProvider;
        }

        private Methodology FetchMethodologyFromFile()
        {
            DataContractJsonSerializer jsonSerializer = new DataContractJsonSerializer(typeof(Methodology));
            Methodology methodology = null;
            try
            {

                using (StreamReader sr = new StreamReader(Environment.CurrentDirectory + "\\Methodology\\MethodologyInfo.json"))
                {
                    object objResponse = jsonSerializer.ReadObject(sr.BaseStream);
                    methodology = objResponse as Methodology;
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("The file could not be read:");
                //Console.WriteLine(exception.Message);
            }
            
            methodology.GeneralInstructions = "General Photo Capture Guidelines" +
                                              "\n" +
                                              "\nIt is recommended the users follow the examples given in this document. General guidelines that apply to all test photo categories are:" +
                                              "\n•  Avoid direct light sources in all scenes" +
                                              "\n•  Avoid moving and animated objects such as cars and people" +
                                              "\n•  Avoid reflective surfaces such as glass or mirrors" +
                                              "\n•  Avoid photographer shadow" +
                                              "\n•  Follow field of view guidelines described in the section below" +
                                              "\n\n" +
                                              "\nField of View Guidelines" +
                                              "\n" +
                                              "\nVIQET can be used to compare the visual quality of images taken from multiple devices. When comparing multiple devices, we recommend that the users:" +
                                              "\n•  Capture the same scenes with all test devices" +
                                              "\n•  Capture as similar a field of view as possible by the test devices." +
                                              "\n" +
                                              "\nWhile capturing the exact field of view is not necessary, we ask that the user ensure that the field of view is as similar as possible by following the following guidelines:" +
                                              "\n•  You might have to move closer or further from the object when switching between devices in order to ensure that a similar field of view is captured." +
                                              "\n•  When capturing photos of objects against a flat background, ensure that the objects are fully captured in the images taken by all test devices." +
                                              "\n•  Make sure you do not add objects into the field of view of one device that are not included in the field of view of another device. ";

            return methodology;
        }
    }
}
