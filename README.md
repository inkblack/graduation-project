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

# 12.1
由于11.15-12.2号主要还是再复习国考所以几乎没有什么进度。

# 12.15
工作内容：找到了python编译器（已上传），开始进行学习。
待编辑。

# 12.30
工作内容：研究接口，先写好了安卓的界面部分。
待编辑。

