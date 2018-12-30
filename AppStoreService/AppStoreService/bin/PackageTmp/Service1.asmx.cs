using System;
using System.Collections;
using System.ComponentModel;
using System.Data;
using System.Web;
using System.Web.Services;
using System.Web.Services.Protocols;

namespace WebService1
{
    /// <summary>
    /// Service1 的摘要说明
    /// </summary>
    [WebService(Namespace = "http://tempuri.org/")]
    [WebServiceBinding(ConformsTo = WsiProfiles.BasicProfile1_1)]
    [ToolboxItem(false)]
    public class Service1 : System.Web.Services.WebService
    {

        [WebMethod]
        public int SendRFIDData(string Temp)
        {
            int ret = 0;
            if (Temp.Length == 0)
            {
                ret= 0;
            }
            else
            {
                ret= 1;
            }
            return ret;
        }
    }
}
