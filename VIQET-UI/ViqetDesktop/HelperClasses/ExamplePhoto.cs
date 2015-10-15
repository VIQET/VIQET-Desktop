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
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace ViqetDesktop
{
    [Serializable()]
    [DataContract]
    public class ExamplePhoto
    {
        [DataMember(Name = "name")]
        public String Name { get; set; }

        [DataMember(Name = "blobFileName")]
        public String FileName { get; set; }

        [DataMember(Name = "description")]
        public String Description { get; set; }

        [DataMember(Name = "acceptable")]
        public bool IsAcceptable { get; set; }

        public BitmapImage Thumbnail
        {
            get { return CommonTasks.FetchImage(this.FileName, 300); }
        }
    }
}
