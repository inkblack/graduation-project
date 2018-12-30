using System;
using System.Collections.Generic;
using System.Web;
using System.Web.Services;

namespace AppStoreService
{
    /// <summary>
    /// Service1 的摘要说明
    /// </summary>
    [WebService(Namespace = "http://tempuri.org/")]
    [WebServiceBinding(ConformsTo = WsiProfiles.BasicProfile1_1)]
    [System.ComponentModel.ToolboxItem(false)]
    public class Service1 : System.Web.Services.WebService
    {

        [WebMethod]
        public string HelloWorld(string userName)
        {
            return "Hello " + userName;
        }
    }
}