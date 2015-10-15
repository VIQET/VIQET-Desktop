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
    public class InputCategoryInfo
    {
        [DataMember(Name = "name")]
        public string Name { get; set; }

        [DataMember(Name = "description")]
        public string Description { get; set; }

        [DataMember(Name = "requiredPhotoCount")]
        public int RequiredPhotoCount { get; set; }

        [DataMember(Name = "examplePhotoList")]
        public List<ExamplePhoto> ExamplePhotoList { get; set; }

        [DataMember(Name = "OutputCategoryName")]
        public List<string> outputCategories { get; set; }
    }
}
