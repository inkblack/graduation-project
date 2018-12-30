<%@ Page Language="C#" AutoEventWireup="true" CodeBehind="Default.aspx.cs" Inherits="AppStoreSite.AppStoreDefault" %>

<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
    <title></title>
</head>
<body>
<div style="text-align:center; width:800px;  ">
    <div style="font-size:32px; font-weight:bold; color:Blue; margin-bottom:20px">
        <span>安卓应用软件商店</span>
    </div>
    <form id="form1" runat="server">
    <div>
        <div style="width:720px; text-align:right ">
            <input id="inputCategory" type="hidden" runat="Server" />
            <asp:TextBox Width="50px" ID="tbSearch" runat="server"></asp:TextBox>
            <asp:Button ID="btnSearch" runat="server" Text="搜索" onclick="btnSearch_Click" />
            <span style="width:10px;" />
            <asp:Button ID="btnUploading" runat="server" Text="我要上传" 
                onclick="btnUploading_Click" />
                
        </div>
        <div style="width:720px; text-align:left ">
            <asp:Button ID="btnAllApp" runat="server" Text="全部" onclick="btnAllApp_Click" />
            <span> >>> </span>
            <asp:Button ID="btnSpecial" runat="server" Text="专题" 
                onclick="btnSpecial_Click" />
            <span>>>></span>
            <asp:Button ID="btnUtility" runat="server" Text="应用" 
                onclick="btnUtility_Click" />
            <span>>>></span>  
            <asp:Button ID="btnAppGame" runat="server" Text="游戏" 
                onclick="btnAppGame_Click" />  
        </div>
        <div style="width:720px; text-align:right ">
            <asp:RadioButton ID="rbByDownloadTimes" runat="server" Text="按下载热度" 
                GroupName="SortTypeGroup" oncheckedchanged="rbByDownloadTimes_CheckedChanged" />
            <asp:RadioButton ID="rbByNewest" runat="server" Text="按发布时间" 
                GroupName="SortTypeGroup" oncheckedchanged="rbByDownloadTimes_CheckedChanged" />
        </div>
        <div style="width:720px; text-align:left; vertical-align:top; height:600px; ">
            <asp:GridView ID="gridView" runat="server" AutoGenerateColumns="False" 
                BackColor="White" BorderColor="#E7E7FF" 
                BorderStyle="None" BorderWidth="1px" CellPadding="3" 
                GridLines="Horizontal" onrowcommand="gridView_RowCommand" ViewStateMode="Enabled" 
                >
                <AlternatingRowStyle BackColor="#F7F7F7" />
                <Columns>
                   
                    <asp:BoundField DataField="app_name" HeaderText="名称" />
                    <asp:BoundField DataField="category" HeaderText="类别" />
                    <asp:BoundField DataField="download_times" HeaderText="下载量" />
                    <asp:BoundField DataField="create_time" HeaderText="上传时间" />
                    <asp:BoundField DataField="file_size" HeaderText="文件大小" />
                    <asp:BoundField />

                    <asp:TemplateField HeaderText="下载">
                        <ItemTemplate>
                            <a target="_blank" href="download.aspx?addr=<%#DataBinder.Eval(Container, "DataItem.relative_path") %>">下载</a>
                            
                        </ItemTemplate>
                    </asp:TemplateField>
                </Columns>
                <FooterStyle BackColor="#B5C7DE" ForeColor="#4A3C8C" />
                <HeaderStyle BackColor="#4A3C8C" Font-Bold="True" ForeColor="#F7F7F7" />
                <PagerStyle BackColor="#E7E7FF" ForeColor="#4A3C8C" HorizontalAlign="Right" />
                <RowStyle BackColor="#E7E7FF" ForeColor="#4A3C8C" />
                <SelectedRowStyle BackColor="#738A9C" Font-Bold="True" ForeColor="#F7F7F7" />
                <SortedAscendingCellStyle BackColor="#F4F4FD" />
                <SortedAscendingHeaderStyle BackColor="#5A4C9D" />
                <SortedDescendingCellStyle BackColor="#D8D8F0" />
                <SortedDescendingHeaderStyle BackColor="#3E3277" />
            </asp:GridView>
        </div>
    </div>
    </form>
</div>
</body>
</html>
