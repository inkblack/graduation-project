using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Services;
using System.IO;
using SQLiteUtility;
using System.Data;
using System.Xml;

namespace AppStoreService
{
    /// <summary>
    /// AppService 的摘要说明
    /// </summary>
    [WebService(Namespace = "http://tempuri.org/")]
    [WebServiceBinding(ConformsTo = WsiProfiles.BasicProfile1_1)]
    [System.ComponentModel.ToolboxItem(false)]
    // 若要允许使用 ASP.NET AJAX 从脚本中调用此 Web 服务，请取消对下行的注释。
    // [System.Web.Script.Services.ScriptService]
    public class AppService : System.Web.Services.WebService
    {

        [WebMethod]
        public string HelloWorld()
        {
            return "Hello World";
        }
        public void CreateSqliteDatabase(string strDBFileName)
        {
            if (strDBFileName.IndexOf(":") < 0)
            {
                strDBFileName = Server.MapPath(strDBFileName);
            }
            SQLiteUtility.SQLiteDBManager.CreateSqliteDatabase(strDBFileName);
        }
        [WebMethod]
        public void InitDatabase()
        {
            String strDir = Server.MapPath("/db");
            if (!Directory.Exists(strDir))
            {
                Directory.CreateDirectory(strDir);
            }
            string strSqliteFile = Server.MapPath("/db/contact.db");
            this.CreateSqliteDatabase(strSqliteFile);
            SQLiteDBManager.SetConnection(strSqliteFile, "");
            string sql;
            sql = "CREATE TABLE contact  ( id INTEGER, name TEXT, mobilephone TEXT, officephone TEXT, familyphone TEXT,";
	        sql += "address TEXT, othercontact TEXT, email TEXT, position TEXT, company TEXT, zipcode TEXT, remark TEXT,";
	        sql += "imageid INTEGER, privacy INTEGER, UserId INTEGER, CONSTRAINT PK_contact PRIMARY KEY (id) )";
            SQLiteDBManager.ExecSqlCmd(sql);
            sql = " CREATE TABLE User ( id INTEGER, UserName TEXT, Password TEXT, remark TEXT, ";
            sql += " CONSTRAINT PK_User PRIMARY KEY (id) )";
            SQLiteDBManager.ExecSqlCmd(sql);

        }
        public bool ConnnectDatabase()
        {
            if (!SQLiteDBManager.OpenConnection())
            {
                string strDBPathName = Server.MapPath("/db/contact.db");
                if (!File.Exists(strDBPathName))
                {
                    InitDatabase();
                }
                SQLiteDBManager.SetConnection(strDBPathName,"");
            }
            return true;
        }
        /// <summary>
        /// app上传保存
        /// </summary>
        /// <param name="userId">用户ID</param>
        /// <param name="contactXml">联系人名单</param>
        /// <returns></returns>
        [WebMethod]
        public int UploadContact(string userId, string contactXml)
        {
            ConnnectDatabase();
            //删除之前的用户联系人列表
            string strDelete = "delete from contact where UserId= " + userId;
            SQLiteDBManager.ExecSqlCmd(strDelete);

            XmlDocument doc = new XmlDocument();
            doc.LoadXml(contactXml);
            XmlNodeList nodeList = doc.SelectSingleNode("persons").ChildNodes;
            int cnt = 0;
            foreach (XmlNode node in nodeList)
            {
                XmlElement xe = (XmlElement)node;
                XmlNodeList xnf1 = xe.ChildNodes;

//                 serializer.startTag(null, "name"); 
//                 serializer.startTag(null, "mobilephone");
//                 serializer.startTag(null, "officephone");
//                 serializer.startTag(null, "familyphone"); 
//                 serializer.startTag(null, "address");
//                 serializer.startTag(null, "othercontact"); 
//                 serializer.startTag(null, "position");
//                 serializer.startTag(null, "company"); 
//                 serializer.startTag(null, "zipcode"); 
//                 serializer.startTag(null, "remark"); 
//                 serializer.startTag(null, "imageid"); 
//                 serializer.startTag(null, "privacy");
                string name, mobile, office, family, address, other, position, company, zipcode, remark, imageid, privacy;
                name= mobile= office= family= address=other= position= company= zipcode= remark= imageid= privacy ="";
                foreach (XmlNode childNode in xnf1)
                {
                    XmlElement chileElement = (XmlElement)childNode;
                    if (chileElement.Name.Equals("name"))
                        name = chileElement.InnerText;
                    else if (chileElement.Name.Equals("mobilephone"))
                        mobile = chileElement.InnerText;
                    else if (chileElement.Name.Equals("officephone"))
                        office = chileElement.InnerText;
                    else if (chileElement.Name.Equals("familyphone"))
                        family = chileElement.InnerText;
                    else if (chileElement.Name.Equals("address"))
                        address = chileElement.InnerText;
                    else if (chileElement.Name.Equals("othercontact"))
                        other = chileElement.InnerText;
                    else if (chileElement.Name.Equals("position"))
                        position = chileElement.InnerText;
                    else if (chileElement.Name.Equals("company"))
                        company = chileElement.InnerText;
                    else if (chileElement.Name.Equals("zipcode"))
                        zipcode = chileElement.InnerText;
                    else if (chileElement.Name.Equals("remark"))
                        remark = chileElement.InnerText;
                    else if (chileElement.Name.Equals("imageid"))
                        imageid = chileElement.InnerText;
                    else if (chileElement.Name.Equals("privacy"))
                        privacy = chileElement.InnerText;
                   
                }
                string sql;
                int id = SQLiteDBManager.GetMaxId("contact", "id");
                sql = "insert into contact(id,UserId) values(" + id.ToString() + "," + userId + ")";
                SQLiteDBManager.ExecSqlCmd(sql);
                sql = "update contact set name = '" + name + "'";
                sql += ",mobilephone='" + mobile + "',officephone='" + office + "',familyphone='" + family + "'";
                sql += ",address='" + address + "',othercontact='" + other + "',position='" + position + "'";
                sql += ",company='" + company + "',zipcode='" + zipcode + "',remark='" + zipcode + "'";
                sql += ",imageid=" + (string.IsNullOrEmpty(imageid) ? "0" : imageid);
                sql += ",privacy=" + (string.IsNullOrEmpty(privacy) ? "0" : privacy);

                sql += " where id = " + id.ToString();

                SQLiteDBManager.ExecSqlCmd(sql);
                cnt++;

            }

            return cnt;
        }
        /// <summary>
        /// 查询指定用户的联系人名单
        /// 
        /// </summary>
        /// <param name="strUserId">用户id</param>
      
        /// <returns></returns>
        [WebMethod]
        public string SearchContact(string strUserId)
        {
            string strXml = "";
            string sql = "select id, name, mobilephone, officephone, familyphone ,";
            sql += "address , othercontact , email , position , company , zipcode , remark ,";
            sql += "imageid , privacy  from contact where UserId=" + strUserId;
           
            ConnnectDatabase();
            System.Data.DataSet ds = SQLiteDBManager.ExecQuery(sql , "contact");
            if (ds != null)
            {
                //ds.Tables[0].WriteXml("d:\\appstores.xml",false);

                XmlDocument doc = new XmlDocument();
                XmlNode root = null;
                XmlNode xmlNode = doc.CreateNode(XmlNodeType.XmlDeclaration, "", "");
                doc.AppendChild(xmlNode);
                root = doc.CreateElement("contact");
                doc.AppendChild(root);
                XmlElement item;
                XmlElement child;
                XmlAttribute attr;
                foreach (DataRow dr in ds.Tables[0].Rows)
                {
                    item = doc.CreateElement("usr");
                    attr = doc.CreateAttribute("id");
                    attr.Value = dr["id"] == DBNull.Value ? "0" : Convert.ToString(dr["id"]);
                    item.Attributes.Append(attr);
                    root.AppendChild(item);

                    child = doc.CreateElement("name");
                    child.InnerText = dr["name"].ToString();
                    item.AppendChild(child);

                    child = doc.CreateElement("mobilephone");
                    child.InnerText = dr["mobilephone"].ToString();
                    item.AppendChild(child);

                    child = doc.CreateElement("officephone");
                    child.InnerText = dr["officephone"].ToString();
                    item.AppendChild(child);
                    
                    child = doc.CreateElement("familyphone");
                    child.InnerText = dr["familyphone"].ToString() ;
                    item.AppendChild(child);

                    child = doc.CreateElement("address");
                    child.InnerText = dr["address"].ToString();
                    item.AppendChild(child);

                    child = doc.CreateElement("othercontact");
                    child.InnerText = dr["othercontact"].ToString();
                    item.AppendChild(child);

                    child = doc.CreateElement("email");
                    child.InnerText = dr["email"].ToString();
                    item.AppendChild(child);

                    child = doc.CreateElement("position");
                    child.InnerText = dr["position"].ToString();
                    item.AppendChild(child);

                    child = doc.CreateElement("company");
                    child.InnerText = dr["company"].ToString();
                    item.AppendChild(child);

                    child = doc.CreateElement("zipcode");
                    child.InnerText = dr["zipcode"].ToString();
                    item.AppendChild(child);

                    child = doc.CreateElement("remark");
                    child.InnerText = dr["remark"].ToString();
                    item.AppendChild(child);
                    child = doc.CreateElement("imageid");
                    child.InnerText = dr["imageid"] == DBNull.Value ? "0" : dr["imageid"].ToString();
                    item.AppendChild(child);

                    child = doc.CreateElement("privacy");
                    child.InnerText =dr["privacy"] == DBNull.Value ? "0" : dr["privacy"].ToString();
                    item.AppendChild(child);
                }
                strXml = doc.InnerXml;
            }
            return strXml;
        }
        /// <summary>
        /// 用户登录
        /// </summary>
        /// <param name="userName"></param>
        /// <param name="pwd"></param>
        /// <returns></returns>
        [WebMethod]
        public int Login(string userName, string pwd)
        {
            int id = -1;
            string sql;
            sql = "select id,password from user where username='" + userName + "'";
            ConnnectDatabase();
            System.Data.DataSet ds = SQLiteDBManager.ExecQuery(sql, "user");
            if (ds == null || ds.Tables[0].Rows.Count < 1)
            {
                return -1;
            }
            string strDBPwd = ds.Tables[0].Rows[0][1].ToString();
            if (!string.Equals(pwd, strDBPwd, StringComparison.CurrentCultureIgnoreCase))
            {
                return -1;
            }
            id = Convert.ToInt32(ds.Tables[0].Rows[0][0]);
            return id;
        }
        [WebMethod]
        public int Regist(string userName, string pwd, string remark)
        {
            int id = 0;
            ConnnectDatabase();
            id = SQLiteDBManager.GetMaxId("user", "id");
            string sql;
            sql = "select id from user where username = '" + userName + "'";
            
            DataSet ds = SQLiteDBManager.ExecQuery(sql, "user");
            bool fNew = true;
            if (ds != null)
            {
                if (ds.Tables[0].Rows.Count > 0)
                {
                    id = Convert.ToInt32(ds.Tables[0].Rows[0][0]);
                    fNew = false;
                }
            }
            if (fNew)
            {
                sql = "insert into user(id,username,password,remark) values(";
                sql += id.ToString() + ",'" + userName + "','" + pwd + "','" + remark + "')";
                if (SQLiteDBManager.ExecSqlCmd(sql) > 0)
                {
                    return id;
                }
                return 0;
            }
            else
            {
                sql = "update user set password='" + pwd + "' where id =" + id.ToString();
                if (SQLiteDBManager.ExecSqlCmd(sql) > 0)
                {
                    return id;
                }
                return 0;

            }
            return 0;
        }
        
    }
}
