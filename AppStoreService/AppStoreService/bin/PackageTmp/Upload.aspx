<%@ Page Language="C#" AutoEventWireup="true" CodeBehind="Upload.aspx.cs" Inherits="AppStoreSite.Upload" %>

<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
    <title></title>
</head>
<body>
    <form id="form1" runat="server">
    <div style="font-size:32px; font-weight:bold; color:Blue; margin-bottom:20px">
    <span>安卓应用软件上传</span>
    </div>
    <div>
        <table style="width:100%;">
            <tr>
                <td style="width:92px;">
                    <div>程序文件:</div>
                    </td>
                <td>
                    
                    <asp:FileUpload ID="FileUpload1" Width="500px" runat="server" />
                    </td>
                <td>
                    </td>
            </tr>
            <tr>
                <td>
                    <div>软件名称：</div>
                </td>
                <td>
                    <asp:TextBox ID="tbAppName" runat="server" Width="200px"></asp:TextBox></td>
                <td></td>
            </tr>
            <tr>
                <td>
                    <div>所属分类：</div>
                </td>
                <td>
                    <asp:DropDownList ID="ddlCategory" runat="server" Width="120px">
                        <asp:ListItem Value="0">专题</asp:ListItem>
                        <asp:ListItem Value="1">应用</asp:ListItem>
                        <asp:ListItem Value="2">游戏</asp:ListItem>
                    </asp:DropDownList>
                </td>
                <td></td>
            </tr>
            <tr>
                <td>
                    <div>软件说明：</div>
                </td>
                <td>
                    <asp:TextBox ID="tbDesc" runat="server" Height="200px" MaxLength="200" 
                        TextMode="MultiLine" Width="700px"></asp:TextBox></td>
                <td></td>
            </tr>
            <tr>
                <td colspan="2" align="center">
                    <asp:Button ID="btnSave" runat="server" Text="保 存" onclick="btnSave_Click" 
                        Width="90px" />
                </td>
            </tr>
            <tr>
                <td colspan="2" align="left" >
                    <asp:Label ID="statusLabel" runat="server" ForeColor="Red"></asp:Label>
                </td>
            </tr>
        </table>
    
        
    
    </div>
    </form>
</body>
</html>
