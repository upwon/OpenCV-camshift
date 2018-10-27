

/// --------------------------------------------------------------------
/// 创建日期：2018/10/26  10:48
/// 创建者：王先文
/// 邮箱：wangxianwenup@outlook.com
/// 所用设备：Windows10 64bit + VisualStudio 2017
/// 更改日期：
/// 概述：OpenCV 彩色目标追踪，参考OpenCV安装目录下..\opencv\sources\samples\cpp\camshiftdemo.cpp\camshiftdemo.cpp
/// --------------------------------------------------------------------



#include <iostream>
//#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
//#include "opencv2/videoio.hpp"
#include<ctype.h>  //C标准函数库头文件，定义了一批C语言字符分类函数
#include "opencv2/highgui.hpp"
/*
highgui为跨平台的gui/IO组件，支持图像、视频、摄像头的文件读取显示以及转码
*/

using namespace std;
using namespace cv;


//声明全局变量




Mat image;
bool backprojMode = false;      //表示是否进入反向投影模式，true表示准备进入反向投影模式
bool selectObject = false;		//代表是否在选要跟踪的初始目标，true表示正在用鼠标选择
int trackObject = 0;			//代表追踪目标数目 ？
bool showHist = true;			//是否显示直方图
Point origin;					//y=用于保存鼠标第一次选择时单击点的位置
Rect selection;					//用于保存鼠标选择的矩形框
int vmin = 10, vmax = 256, smin = 30;

//onMouse鼠标回调函数

static void onMouse(int event,int x,int y,int,void*)
{
	/*
		鼠标移动时触发if(selectObject)
		然后新的坐标点的x,y值都会传过来
	
	*/
	if(selectObject)
	{
		selection.x = MIN(x, origin.x);        //矩形左上角顶点坐标
		selection.y = MIN(y, origin.y);
		selection.width = std::abs(x - origin.x);  //矩形宽
		selection.height = std::abs(y - origin.y); //矩形高
	
		//用于确定所选的矩形区域在图片范围内 ？
		selection &= Rect(0, 0, image.cols, image.rows);
	
	}

	switch (event)
	{
	 //鼠标按下去是一个事件，传到这个函数里面，触发case CV_EVENT_LBUTTONDOWN: 这一行 
	case CV_EVENT_LBUTTONDOWN:
		origin = Point(x, y);
		selection = Rect(x, y, 0, 0);
		selectObject = true;    //选在跟踪时的初始目标
		break;

	 //鼠标左键抬起这个事件传到函数里，触发case CV_EVENT_LBUTTONUP:这一行  

	case CV_EVENT_LBUTTONUP:
		selectObject = false;   //不选在跟踪时的初始目标
		if (selection.width >0 && selection.height > 0 )
			trackObject = -1;
		break;

	default:
		break;
	}



}


//help( )函数  输出帮助信息

static void ShowHelpText()
{
	cout<<"\n\n\t基于均值漂移的追踪（tracking）技术\n"
		"\t请用鼠标框选一个有颜色的物体，对它进行追踪操作\n"; 


	cout << "\n\n\t操作说明： \n"
		"\t\t用鼠标框选对象来初始化跟踪\n"
		"\t\tESC - 退出程序\n"
		"\t\tc - 停止追踪\n"
		"\t\tb - 开/关-投影视图\n"
		"\t\th - 显示/隐藏-对象直方图\n"
		"\t\tp - 暂停视频\n";


}
 
const char* keys =
{
	"{1|  | 0 |  camera number}"

};



//main()函数  控制台应用程序入口
int main()
{
	ShowHelpText(); 

	VideoCapture cap;   //定义一个摄像头捕获的类对象
	Rect trackWindow;
	int hsize = 16;
	float hranges[] = { 0,180 };     //直方图的范围  hranges在后面计算直方图的函数中用到
	const float* phranges = hranges;


	cap.open(0);     //调用成员函数打开摄像头

	if (!cap.isOpened())
	{
		cout << "不能初始化摄像头\n";

	}

	namedWindow("直方图",0);
	namedWindow("CamShift", 0);
	setMouseCallback("CamShift",onMouse,0);  //消息响应机制

	//createTrackbar函数的功能是在对应的窗口创建滑动条，
	createTrackbar("Vmin","CamShift",&vmin,256,0);		// 滑动条Vmin, vmin表示滑动条的值，最大为256
	createTrackbar("Vmax", "CamShift", &vmax, 256, 0);  //最后一个参数为0代表没有调用滑动拖动的响应函数  
	createTrackbar("Smin","CamShift",&smin,256,0);		//vmin,vmax,smin初始值分别为10,256,30  

	/*
		CV_8UC1，CV_8UC2，CV_8UC3。最后的1、2、3表示通道数，譬如RGB3通道就用CV_8UC3。
		CV_8UC3 表示使用8位的 unsigned char 型，每个像素由三个元素组成三通道,
		初始化为（0，0，255）
		http://blog.csdn.net/augusdi/article/details/8876459

		Mat::zeros 返回指定的大小和类型的零数组。
		C++: static MatExpr Mat::zeros(int rows, int cols, int type)
		rows–行数。    cols  –列数。type– 创建的矩阵的类型。
		A = Mat::zeros （3，3，CV_32F）；
		在上面的示例中，只要A不是 3 x 3浮点矩阵它就会被分配新的矩阵。
		否则为现有的矩阵 A填充零。
		转自:http://blog.csdn.net/sherrmoo/article/details/40951997

		hist 直方图数字矩阵 最后 -> histimg 直方图图像
		hsv ->（取出h） hue
		mask？ 掩膜？  --------------------------------------------------------------------------------------------------？？？？？
		backproj 反向投影的矩阵
	*/


	Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 300, CV_8UC3), backppoj;
	bool paused = false;

	for (;;)
	{
		if (!paused)    //没有暂停
		{
			cap >> frame;       //从摄像头抓取一帧图像并输入到frame中
			if (frame.empty())
				break;
		}

		frame.copyTo(image);

		if (!paused)  //没有按键暂停
		{
			cvtColor(image, hsv, CV_BGR2HSV);      //将rgb摄像头帧转化成hsv空间的---转hsv

			 //trackObject初始化为0,或者按完键盘的'c'键后也为0，当鼠标单击松开后为-1 
			if (trackObject)
			{
				int _vmin = vmin, _vmax = vmax;
				/*
				inRange函数的功能是检查输入数组每个元素大小是否在2个给定数值之间，
				可以有多通道,mask保存0通道的最小值，也就是h分量
				这里利用了hsv的3个通道，
				比较h,0~180,s,smin~256,v,min(vmin,vmax),max(vmin,vmax)。
				如果3个通道都在对应的范围内，则
				mask对应的那个点的值全为1(0xff)，否则为0(0x00).
				*/

				inRange(hsv, Scalar(0, smin, MIN(_vmin, _vmax)), Scalar(180, 256, MAX(_vmin, _vmax)), mask);
				int ch[] = { 0,0 };

				//hue初始化为与hsv大小深度一样的矩阵，色调的度量是用角度表示的
				//红绿蓝之间相差120度，反色相差180度
				hue.create(hsv.size(), hsv.depth());          //HSV(Hue,saturation,Value)

				/*
				int from_to[] = { 0,2, 1,1, 2,0, 3,3 };
				mixChannels( &rgba, 1, out, 2, from_to, 4 );
				from_to:通道交换对，数组中每两个元素为一对，表示对应的交换通道
				pair_count：通道交换对个数（即*from_to数组中行数）
				http://blog.163.com/jinlong_zhou_cool/blog/static/22511507320138932215239/

				bgr 三原色RGB混合能形成其他的颜色，所以不能用一个值来表示颜色
				hsv H是色彩  S是深浅,S = 0时，只有灰度  V是明暗
				hsv -> hue 把色彩单独分出来
				http://blog.csdn.net/viewcode/article/details/8203728
				*/

				mixChannels(&hsv, 1, &hue, 1, ch, 1);
				if (trackObject < 0)    //鼠标选择区域松开后，该函数内部又将其赋值-1
				{
					//此处的构造函数roi用的是Mat hue的矩阵头
					//且roi的数据指针指向hue，即共用相同的数据，select为其感兴趣的区域
					Mat roi(hue,selection),maskroi(mask,selection);		//mask保存的hsv的最小值
					 /*
					将roi的0通道计算直方图并通过mask放入hist中，hsize为每一维直方图的大小
					calcHist函数来计算图像直方图
					---calcHist函数调用形式
					C++: void calcHist(const Mat* images, int nimages, const int* channels,
					InputArray mask, OutputArray hist, int dims, const int* histSize,
					const float** ranges, bool uniform=true, bool accumulate=false
					参数详解
					onst Mat* images：输入图像
					int nimages：输入图像的个数
					const int* channels：需要统计直方图的第几通道
					InputArray mask：掩膜，，计算掩膜内的直方图  ...Mat()
					OutputArray hist:输出的直方图数组
					int dims：需要统计直方图通道的个数
					const int* histSize：指的是直方图分成多少个区间，，，就是 bin的个数
					const float** ranges： 统计像素值得区间
					bool uniform=true::是否对得到的直方图数组进行归一化处理
					bool accumulate=false：在多个图像时，是否累计计算像素值得个数
					http://blog.csdn.net/qq_18343569/article/details/48027639
					*/


				}
			}



		}
	}

}
