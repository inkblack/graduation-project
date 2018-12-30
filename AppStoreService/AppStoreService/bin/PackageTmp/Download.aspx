<%@ Page Language="C#" AutoEventWireup="true" CodeBehind="Download.aspx.cs" Inherits="AppStoreSite.Download" %>

<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
    <title></title>
</head>
<body>
<div style="text-align:center; width:800px;  ">
    <div style="font-size:32px; font-weight:bold; color:Blue; margin-bottom:20px">
        <span>安卓应用软件商店-软件下载</span>
    </div>
    <form id="form1" runat="server">
    <div style=" height:400px; width:400px; vertical-align:middle; text-align:center;  ">
        <asp:Button ID="btnDownload" runat="server" Text="下载" 
            onclick="btnDownload_Click" />
    </div>
    </form>
</div>
</body>
</html>
