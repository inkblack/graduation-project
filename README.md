# 毕业设计


## 项目内容的功能描述


本项目功能目前确定是能够实现在android平台上进行python语言程序的编译。
目前类似的软件如SL4A有着文档少且效率低的问题，并且也不能和java进行交互。本项目试图能够完成一个可以在安卓手机上进行流畅、效率地python编写、编译的过程，在完成基本功能的情况下，尽可能实现和java的交互。

## 项目参考文献

目前还处于对整个项目构建的了解中，所以具体技术还有待补充，以下先列出参考资料

[overflow上对于这个问题的讨论]
[在PC上的交叉编译android的python compiler]


[overflow上对于这个问题的讨论]:https://stackoverflow.com/questions/101754/is-there-a-way-to-run-python-on-android

[在PC上的交叉编译android的python compiler]:https://mdqinc.com/blog/2011/09/cross-compiling-python-for-android/


## 时间安排


11.2-11.15 查阅参考资料并加以学习。

11.16-12.~ 进行代码初步的编写（这个过程不知道要持续多久，可能要占去12月的大部分时间）

预计12月内完成，1月主要是DEBUG






# 会议部分

## 第一次会议
本人阅读的两篇论文分别为：


**Augmented Vehicular Reality: Enabling Extended Vision for Future Vehicles**

本文陈述了一种汽车视觉增强的新的可能性，在仅使用立体相机的低成本下，通过画面中的静态景物建立坐标轴，同步两辆汽车的视觉，以此达到拓展汽车视野的目的。为自动驾驶、辅助驾驶提出了全新的可能性。

**Depth Aware Finger Tapping on Virtual Displays**

本文主要是实现了一种使用基于轻量级超声波的传感，以及一个商用现货（COTS）单镜头，以实现用户手指的3D跟踪的方法。通过超声波反射确定手指的移动、移动幅度、弯曲程度，同时使用视频确认是否真的移动，确认移动的手指。通过两种方法结合，大大减少了用户手指移动的误判。

同时提出了第一次的毕设构想：做一个秘书型的日历，被否决。


## 第二次会议
本次会议主要是剩下几个同学做presentation，讲了几篇他们关注的论文，并讨论了还未确定题目的同学的毕业设计项目，同时确定了所有同学的毕业设计题目。我自己提出做一个能够在android上运行python的软件，老师给出要求是首先完成到一个类似Termux的状况，如果可以，实现能够直接在软件中和java交互的功能。目前感觉相当有难度……

# 第一、二周工作总结

按照先前的安排，头两周我主要做的事情还是进行资料的查找和学习。目前还没有开始进行代码编译的工作。
首先是关于Python编译器的相关内容，除了官网之外，在国内的博客找到一个关于python发行版（编译器）相关的详细内容介绍：http://www.cnblogs.com/mehome/p/9427229.html

另外在寻找python编译器开源代码的同时，找到了一个在线编译器的简单原理与实现：https://blog.csdn.net/u013055678/article/details/73896477

以及本文中提到的在线编译器：
python3.0:  http://www.runoob.com/try/runcode.php?filename=HelloWorld&type=python3

python2.0:  http://www.runoob.com/try/runcode.php?filename=HelloWorld&type=python

带窗口的python3.0: http://www.dooccn.com/python3/

考虑到我要实现的，感觉应该会有帮助。

再加上考虑到有可能会使用C或C++来进行主要功能的实现，所以对Android NDK做了一定程度的了解（这个还在了解的过程中），主要是通过官网和相关内容的博客（数量很多此处不方便一一贴出地址，只给出官网）进行学习，官网地址：https://developer.android.google.cn/ndk/

因为报名了12月2号的国考所以目前进度不是很快，估计大部分的代码编写会集中在12月内进行。

## 12月到1月开发记录

# 设计思路
使用C ++语言进行对python语言的解析，然后通过JNI进行JAVA和C ++之间的通信
JAVA部分就是界面的实现，这一部分会先进行。设计界面比较简单，分成两部分，上半部分是代码输入，下半部分是结果输出。

编译器的实现部分
lexical：进行词义分析
parse：进行语法分析和运行
mypython：编译器主体，调用上面两个函数
lib-native:JNI部分的实现

# 具体实现
使用Android Studio进行编写。先完成界面的部分，简单把界面分为上下输入和输出两个部分。

完成JAVA部分手机存储权限的获取和文件的读入和写出部分。

完成编译器部分的lexical词义分析部分

设计getChar()和addChar()来进行字符获取，读取用户输入的每一个字符并写入对应vector类，函数lex()来进行对词义的判断（变量、数字或是符号），运算符单独进行记录。

parser部分没有进行具体的内容编写。

完成了mypython部分代码，方便进行JNI的编写。JNI中主要函数为
extern "C" JNIEXPORT jstring JNICALL
Java_com_demo_python4android_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject , jstring fileName)
返回jstring，方便在JAVA页进行输出。

## 2月的开发过程

# 具体实现内容

完成parser的内容。

parser中定义变量、函数类来辅助进行后续编写。

三个类中，函数继承自变量类，因为函数名同样可以作为一个变量来看，为此需要单独写一个函数variableLookup来确定变量究竟是否为函数。

在parser中需要定义各种运算规则。最基础的是给函数赋值，然后是加减乘除的四则运算，由于运算级优先度不同，所有加减和乘除分开判断。此外还有对于布尔表达式的定义（为了进行条件判断），左右括号、引号的规则，简单的函数定义以及调用。

在这部分开发中，由于条件判断、函数定义都需要确认某一个条件作用的定义域范围，所有定义一个定义域类来确定。确定定义域的方法，在多次尝试之后选择使用对于缩进空格的判断。因为按照标准格式输入的代码中，条件判断和函数的具体内容都有缩进。

所以这里选择在lexical中添加一个类单独作为空格、换行和缩进的判断。在parser中就可以更加方便的确认作用域了。

## 3月的开发过程

# 具体内容

完善了mypython中对于parser和lexical的内容，软件已经可以运行。

进行具体的测试和DEBUG。

主要进行的还是parser的DEBUG，因为要保证每一个运算都正确，所以主要是进行大量的测试来进行DEBUG。出错了就一点点倒回去修改。

修改了parser中进行运算的主要函数evaluate，从单个栈变为使用两个栈来进行实现，之前用单个栈经常会出现运算顺序不正确的情况，主要还是因为局限性很大，导致当括号出现位置、乘除号位置改变的时候不能完全成功。

改成两个栈之后，一个用于获取当前的vector，将运算符作为运算数字的后缀放进去，然后通过switch选择对于不同运算级进行分类判断后，推到栈2中，确保完全是按照正确的优先级进行计算。这样之后四则运算的正确率提升，已经基本保证不出现BUG的情况了。

修改了对于布尔表达式的判断，确保了括号出现的正确的位置。


## 4月的开发过程

主要进行文献的查找和论文的编写。



