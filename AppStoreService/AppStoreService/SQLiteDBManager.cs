using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SQLite;
using System.Net;

namespace SQLiteUtility
{//sql = "update Product set Name=:name , Brand=:brand ,PlaceOfOrigin=:place ,Model =:model,Standard=:standard,ProductLine=:pl ";
    public class SQLiteDBManager
    {
        public static System.Data.SQLite.SQLiteConnection m_Conn=null;
        //Provider=Microsoft.Jet.OLEDB.4.0;Data Source=E:\work\SaleStatistics\SaleStatistics\bin\Debug\db.mdb;Persist Security Info=True;Jet OLEDB:Database Password=8954762
        public static string g_StrConn = @"Provider=Microsoft.Jet.OLEDB.4.0;Data Source=.\db.mdb;Persist Security Info=True;Jet OLEDB:Database Password=8954762";		//服务器连接字符串



        public SQLiteDBManager()
		{
		}
        public static void CreateSqliteDatabase(String strDBFileName)
        {
            SQLiteConnection.CreateFile(strDBFileName);
        }
        public static bool SetDatabasePassword(string pwd)
        {
            if (m_Conn == null)
                return false;
            try
            {
                if (m_Conn.State != ConnectionState.Open)
                {
                    m_Conn.Open();
                }
                m_Conn.ChangePassword(pwd);
                return true;
            }
            catch (System.Exception ex)
            {
                return false;
            }
            

        }
		public static  void SetConnection(string strConn)
		{
            //Data Source=PoliceEquipment.db;Version=3;Password=C airn^8971#2067
			if(m_Conn != null)
			{
				if(m_Conn.State == System.Data.ConnectionState.Open)
				{
					m_Conn.Close();
				}
			}
            m_Conn = new SQLiteConnection(strConn);
		}
        public static void SetConnection(string dbfile, string password)
        {
            string strConn = "Data Source=" + dbfile;
            strConn += ";Version=3";
            if (!string.IsNullOrEmpty(password))
            {
                strConn += ";Password=" + password;
            }
            if (m_Conn != null)
            {
                if (m_Conn.State == System.Data.ConnectionState.Open)
                {
                    m_Conn.Close();
                }
            }
            m_Conn = new SQLiteConnection(strConn);
        }
        public static bool OpenConnection()
        {
            if (m_Conn == null)
                return false;
            try
            {
                if (m_Conn.State == ConnectionState.Open)
                    return true;
                m_Conn.Open();
                return true;
            }
            catch (System.Exception ex)
            {
                throw ex;
            }
        }
		//执行查询
		public static DataSet ExecQuery(string sql,string table_name)
		{

			if(m_Conn == null)
			{
				return null;
			}
			if(m_Conn.State != System.Data.ConnectionState.Open)
			{
				m_Conn.Open();
			}
			System.Data.SQLite.SQLiteCommand cmd;
			System.Data.SQLite.SQLiteDataAdapter ada;
			System.Data.DataSet ds=new DataSet();
            cmd = new SQLiteCommand();
			cmd.CommandText= sql;
			cmd.CommandType = CommandType.Text;
			cmd.Connection = m_Conn;
            ada = new SQLiteDataAdapter(cmd);
			try
			{
				ada.Fill(ds,table_name);
			}
			catch(System.Data.OleDb.OleDbException e)
			{
                throw (e);			
			}
			return ds;
		}

        public static DataSet ExecQuery(string sql, string table_name,List<SQLiteParameter> ps)
        {

            if (m_Conn == null)
            {
                return null;
            }
            if (m_Conn.State != System.Data.ConnectionState.Open)
            {
                m_Conn.Open();
            }
            SQLiteCommand cmd;
            SQLiteDataAdapter ada;
            System.Data.DataSet ds = new DataSet();
            cmd = new SQLiteCommand();
            cmd.CommandText = sql;
            cmd.CommandType = CommandType.Text;
            cmd.Connection = m_Conn;
            if (ps != null)
            {
                foreach (SQLiteParameter param in ps)
                {
                    cmd.Parameters.Add(param);
                }
            }
            ada = new SQLiteDataAdapter(cmd);
            try
            {
                ada.Fill(ds, table_name);
            }
            catch (Exception e)
            {
                throw (e);
            }
            return ds;
        }
		//执行sql命令
		public static int  ExecSqlCmd(string sql)
		{
			if(m_Conn == null)
			{
				return -1;
			}
			if(m_Conn.State != System.Data.ConnectionState.Open)
			{
				m_Conn.Open();
			}

            SQLiteTransaction trans;
			trans = m_Conn.BeginTransaction(System.Data.IsolationLevel.ReadCommitted);
			SQLiteCommand cmd;
            cmd = new SQLiteCommand();
			cmd.Connection = m_Conn;
			
			cmd.CommandText = sql;	
			cmd.Transaction = trans;
			
			int res =-1;
			try
			{
				
				res = cmd.ExecuteNonQuery();
				trans.Commit();				
			}
			catch(Exception e)
			{
				trans.Rollback();
                throw (e);
			}
			finally
			{
				m_Conn.Close();
			}
			return res;
		}
        public static int ExecSqlCmd(string sql, List<SQLiteParameter> ps)
        {
            if (m_Conn == null)
            {
                return -1;
            }
            if (m_Conn.State != System.Data.ConnectionState.Open)
            {
                m_Conn.Open();
            }
            SQLiteTransaction trans;
            trans = m_Conn.BeginTransaction(System.Data.IsolationLevel.ReadCommitted);
            SQLiteCommand cmd;
            cmd = new SQLiteCommand();
            cmd.Connection = m_Conn;
            cmd.CommandText = sql;
            if(ps != null)
            {
                foreach (SQLiteParameter param in ps )
                {
                    cmd.Parameters.Add(param);
                }
            }
            cmd.Transaction = trans;

            int res = -1;
            try
            {

                res = cmd.ExecuteNonQuery();
                trans.Commit();
            }
            catch (Exception e)
            {
                trans.Rollback();
                Console.WriteLine(e.Message);
                throw (e);
            }
            finally
            {
                m_Conn.Close();
            }
            return res;
        }
		
		//获取最大序号
		public static int GetMaxId(string talbe_name,string field_name)
		{
            int id = 0;
			string sql;
				
			sql = "select max(" + field_name + ") as id from " + talbe_name  ;
            System.Data.DataSet ds = SQLiteDBManager.ExecQuery(sql, "maxid");	
			if(ds != null)
			{
				if(ds.Tables.Count > 0)
				{
					if(ds.Tables[0].Rows.Count > 0)
					{
                        if (ds.Tables[0].Rows[0][0] != DBNull.Value)
                            id = Convert.ToInt32(ds.Tables[0].Rows[0][0] == DBNull.Value ? 0 : ds.Tables[0].Rows[0][0]);
					}
				}
			}
            id++;
			return id;
		}

// 		public static string GetLocalIP()
// 		{
// 			string hostName = System.Net.Dns.GetHostName();
// 			System.Net.IPHostEntry entry = System.Net.Dns.GetHostByName(hostName);
// 			System.Net.IPAddress addr = entry.AddressList[0];
// 			string ip = addr.ToString();
// 			return ip;
// 		}
    }
}
