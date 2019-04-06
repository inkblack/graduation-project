#include <jni.h>
#include <string>
#include "mypython.h"

extern std::string outputStream;

//JNIEXPORT 输出(返回)类型 JINCALL Java _ 类引用路径 _ 方法名

extern "C" JNIEXPORT jstring JNICALL
Java_com_demo_python4android_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */,jstring fileName) {

     const char* str;
     str = env->GetStringUTFChars(fileName, NULL);
     if(str == NULL) {
     return NULL;
     }
     //释放资源
     env->ReleaseStringUTFChars(fileName, str);
    int ret=run(str);
    std::string hello="run error";
    if(ret==0){
    hello=outputStream;
    }
    return env->NewStringUTF(hello.c_str());
}
