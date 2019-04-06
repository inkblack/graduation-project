package com.demo.python4android;

import android.Manifest;
import android.app.NativeActivity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.UnsupportedEncodingException;

public class MainActivity extends AppCompatActivity {

    //允许读取内存
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE};

    // Used to load the 'native-lib' library on application startup.加载本机的lib库
    static {
        System.loadLibrary("native-lib");
    }

    private int REQUEST_PERMISSION_CODE = 100;
    private int REQUEST_OPENFILE_CODE = 101;
    private EditText tvSrc;
    private TextView tvLog;
    private String path;
    private String content;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.LOLLIPOP) {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, PERMISSIONS_STORAGE, REQUEST_PERMISSION_CODE);
            }
        }

        findViewById(R.id.btn_open).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //调用系统文件管理器打开指定路径目录
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("file/*.py");
                intent.addCategory(Intent.CATEGORY_OPENABLE);
                startActivityForResult(intent, REQUEST_OPENFILE_CODE);
            }
        });


        findViewById(R.id.btn_run).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //运行执行，读写内存
                try {
                    String sdCardDir = Environment.getExternalStorageDirectory().getAbsolutePath();
                    String path=sdCardDir+"/python4android";
                    File file=new File(path);
                    if(!file.exists())
                        file.mkdir();

                    sdCardDir=path+"/tmp.py";
                    File saveFile = new File(sdCardDir);
                    FileOutputStream outStream = new FileOutputStream(saveFile);
                    outStream.write(tvSrc.getText().toString().getBytes());
                    outStream.close();
                    //运行部分
                    tvLog.setText(stringFromJNI(sdCardDir));
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });


        // 调用本机方法eg
        tvSrc = findViewById(R.id.tv_source);
        tvLog = findViewById(R.id.tv_log);

        if (Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED)) {
            String fileName = Environment.getExternalStorageDirectory().getAbsolutePath() + "/test.py";

        }
    }

    //回调
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        boolean isSucc = true;
        if (requestCode == REQUEST_PERMISSION_CODE) {
            for (int i = 0; i < permissions.length; i++) {
                Log.i("MainActivity", "申请的权限为：" + permissions[i] + ",申请结果：" + grantResults[i]);
                if (grantResults[i] != 0) {
                    isSucc = false;
                    break;
                }
            }
        }
        if (!isSucc) {
            ActivityCompat.requestPermissions(this, PERMISSIONS_STORAGE, REQUEST_PERMISSION_CODE);
        }
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        //回调函数
        if (requestCode == REQUEST_OPENFILE_CODE) {
            Uri uri = data.getData();
            if ("file".equalsIgnoreCase(uri.getScheme())) {//使用第三方应用打开
                path = uri.getPath();
                return;
            }
            path = getPath(uri);
            tvLog.setText("打开文件:" + path + "\r\n");

            String encoding = "UTF-8";
            File file = new File(path);
            Long filelength = file.length();
            byte[] filecontent = new byte[filelength.intValue()];
            try {
                FileInputStream in = new FileInputStream(file);
                in.read(filecontent);
                in.close();
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
            try {
                content= new String(filecontent, encoding);
                tvSrc.setText(content);
            } catch (UnsupportedEncodingException e) {
                e.printStackTrace();
            }

        }
    }

    public String getPath(Uri uri) {
        //uri转换为绝对路径
        if ("content".equalsIgnoreCase(uri.getScheme())) {
            String[] projection = {"_data"};
            Cursor cursor = null;
            try {
                cursor = getContentResolver().query(uri, projection, null, null, null);
                int column_index = cursor.getColumnIndexOrThrow("_data");
                if (cursor.moveToFirst()) {
                    return cursor.getString(column_index);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else if ("file".equalsIgnoreCase(uri.getScheme())) {
            return uri.getPath();
        }
        return null;
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI(String fileName);
}
