/*
Copyright © 2015 Intel Corporation
This program and the accompanying materials are made available under the terms of the Eclipse Public License v1.0, 
 which accompanies this distribution, and is available at http://www.eclipse.org/legal/epl-v10.html . https://github.com/viqet
* Contributors:
*    Intel Corporation - initial API and implementation and/or initial documentation
*/
using System;
using System.Collections.Generic;
using System.Runtime.Serialization;


namespace ViqetDesktop
{
    [Serializable()]
    [DataContract]
    public class Methodology
    {
        [DataMember(Name = "Version")]
        public string Version { get; set; }

        [DataMember(Name = "inputCategories")]
        private List<InputCategoryInfo> InputCategoryList;
        public Dictionary<string, InputCategoryInfo> InputCategories
        {
            get
            {
                Dictionary<String, InputCategoryInfo> inputCategoryDictionary = new Dictionary<String, InputCategoryInfo>();
                foreach (InputCategoryInfo category in InputCategoryList)
                {
                    inputCategoryDictionary.Add(category.Name, category);
                }
                return inputCategoryDictionary;
            }
        }

        [DataMember(Name = "outputCategories")]
        private List<OutputCategoryInfo> OutputCategoryList;
        public Dictionary<string, OutputCategoryInfo> OutputCategories
        {
            get
            {
                Dictionary<String, OutputCategoryInfo> outputCategoryDictionary = new Dictionary<String, OutputCategoryInfo>();
                foreach (OutputCategoryInfo category in OutputCategoryList)
                {
                    outputCategoryDictionary.Add(category.Name, category);
                }
                return outputCategoryDictionary;
            }
        }
        
        [IgnoreDataMember]
        public string GeneralInstructions { get; set; }
        
    }
}
