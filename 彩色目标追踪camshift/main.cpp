

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

					calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
					/*
					将hist矩阵进行数组范围归一化，都归一化到0~255
					void normalize(const InputArray src, OutputArray dst, double alpha=1,
					double beta=0,
					int normType=NORM_L2, int rtype=-1, InputArray mask=noArray())
					当用于归一化时，normType应该为cv::NORM_MINMAX，alpha为归一化后的最大值，
					beta为归一化后的最小值
					http://www.cnblogs.com/mikewolf2002/archive/2012/10/24/2736504.html
					*/

					normalize(hist,hist,0,255,CV_MINMAX);
					trackWindow = selection;

					/*
					只要鼠标选完区域松开后，且没有按键盘清0键'c'，则trackObject一直保持为1，
					因此该if函数  if( trackObject < 0 )  只能执行一次，除非重新选择跟踪区域。
					*/

					trackObject = 1;

					/*
					与按下'c'键是一样的，这里的all(0)表示的是标量 全部清0

					inline CvScalar cvScalarAll( double val0123 );
					同时设定VAL0,1,2,3的值；
					OpenCV里的Scalar：all的意思：
					scalar所有元素设置为0，其实可以scalar::all(n)，就是原来的CvScalarAll(n)；
					*/

					histimg = Scalar::all(0);

					/*
					histing是一个200*300的矩阵，hsize应该是每一个bin的宽度，
					也就是histing矩阵能分出几个bin出来
					opencv直方图的bins中存储的是什么?
					https://zhidao.baidu.com/question/337997654.html

					假设 有数值 0,0,1,2,3,10,12，13 。
					你分的bins为 0-6 为第一个bin，7-13 为一个bins。
					那么bins[0] 即第一个bins 存储的数就是 4，
					原因是 0,0,1,2,3在第一个bin的范围内，
					bins[1] 存储的数为 3，原因是 10,12,13落在这个[7-13]这个bin内。

					Line111 : hsize=16
					*/

					int binW = histimg.cols / hsize;	//算出宽

					/*
						Mat::Mat(); //default
						Mat::Mat(int rows, int cols, int type);
						Mat::Mat(Size size, int type);
						Mat::Mat(int rows, int cols, int type, const Scalar& s);
					参数说明：
						int rows：高
						int cols：宽
						int type：参见"Mat类型定义"
						Size size：矩阵尺寸，注意宽和高的顺序：Size(cols, rows)
						const Scalar& s：用于初始化矩阵元素的数值
						const Mat& m：拷贝m的矩阵头给新的Mat对象，
						但是不复制数据！相当于创建了m的一个引用对象
					转自：http://blog.csdn.net/holybin/article/details/17751063
					定义一个缓冲单bin矩阵。这里使用的是第二个 重载 函数。
					重载函数：https://baike.baidu.com/item/%E9%87%8D%E8%BD%BD%E5%87%BD%E6%95%B0/3280477?fr=aladdin
					*/

					Mat buf(1, hsize, CV_8UC3);

					/*
					saturate_cast函数为从一个初始类型准确变换到另一个初始类型
					saturate_cast<uchar>(int v)的作用 就是防止数据溢出，
					具体的原理可以大致描述如下：
					if(data<0)
							data=0;
					if(data>255)
					data=255

					转自:http://blog.csdn.net/wenhao_ir/article/details/51545330?locationNum=10&fps=1
					Vec3b为3个char值的向量
					CV_8UC3 表示使用8位的 unsigned char 型，每个像素由三个元素组成三通道,
					初始化为（0，0，255）
					*/

					for (int i = 0;  i < hsize; i++)
					{
						/*
					   互补色相差180度
					   颜色->hsv->hue(0,255)->roi->hist(0,255)

					   所以这里只是，以i为输入，把直方图本来各个矩形的颜色算出来，放在buf里。
					   hsv三个值的取值范围:
					   h 0-180
					   s 0-255
					   v 0-255
					   http://blog.csdn.net/taily_duan/article/details/51506776
					   https://wenku.baidu.com/view/eb2d600dbb68a98271fefadc.html
					   */

						buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180. / hsize), 255, 255);
						cvtColor(buf, buf, CV_HSV2BGR);   //将hsv又转换成bgr，画矩形颜色的用BGR格式

					}

					for (int i = 0; i < hsize ;i++)
					{
						/*
					   at函数为返回一个指定数组元素的参考值
					   histimg.rows常量=200
					   val决定各个矩形的高度
					   */
						int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows / 255);


						/*
					   C++: void rectangle(Mat& img, Rect rec, const Scalar& color, int thickness=1, int lineType=8, int shift=0 )
					   参数介绍：
					   img 图像.
					   pt1 矩形的一个顶点。
					   pt2 矩形对角线上的另一个顶点
					   color 线条颜色 (RGB) 或亮度（灰度图像 ）(grayscale image）。
					   thickness 组成矩形的线条的粗细程度。取负值时（如 CV_FILLED）
					   函数绘制填充了色彩的矩形。
					   line_type 线条的类型。见cvLine的描述
					   https://zhidao.baidu.com/question/427970238676959132.html

					   shift 坐标点的小数点位数。

					   histimg.rows是一个常量。histmig.rows=200
					   Scalar(buf.at<Vec3b>(i))  buf是颜色
					   计算机里，坐标在左上角，x轴朝右，y朝下
					   val决定各个矩形的高度
					   */
						rectangle(histimg, Point(i*binW, histimg.rows), Point((i + 1)*binW, histimg.rows - val), Scalar(buf.at<Vec3b>(i)), -1, 8);

					}





				}
				/*
				反向投影 
				http://blog.csdn.net/qq_18343569/article/details/48028065
				*/
				calcBackProject(&hue, 1, 0, hist, backppoj, &phranges);
				//相与??????这一句注释了也没事 = = ------加上的话，整体来说，噪音要少很多

				backppoj &= mask;

				/*
				Camshift它是MeanShift算法(Mean Shift算法，又称为均值漂移算法)的改进，
				称为连续自适应的MeanShift算法，
				CamShift算法的全称是"Continuously Adaptive Mean-SHIFT"，
				它的基本思想是视频图像的所有帧作MeanShift运算，并将上一帧的结果
				（即Search Window的中心和大小）
				作为下一帧MeanShift算法的Search Window的初始值，如此迭代下去。


				对于OPENCV中的CAMSHIFT例子，是通过计算目标HSV空间下的HUE分量直方图，
				通过直方图反向投影得到目标像素的概率分布，
				然后通过调用CV库中的CAMSHIFT算法，自动跟踪并调整目标窗口的中心位置与大小。
				https://baike.baidu.com/item/Camshift/5302311?fr=aladdin

				cvCamShift(IplImage* imgprob, CvRect windowIn, CvTermCriteria criteria,
				CvConnectedComp* out, CvBox2D* box=0);
				imgprob：色彩概率分布图像。
				windowIn：Search Window的初始值。
				Criteria：用来判断搜寻是否停止的一个标准。
				out：保存运算结果,包括新的Search Window的位置和面积。
				box：包含被跟踪物体的最小矩形。
				http://blog.csdn.net/houdy/article/details/191828

				CV_INLINE  CvTermCriteria  cvTermCriteria( int type, int max_iter,
				double epsilon )
				{
				CvTermCriteria t;
				t.type = type;
				t.max_iter = max_iter;
				t.epsilon = (float)epsilon;
				return t;
				}
				该函数是内联函数，返回的值为CvTermCriteria结构体。
				看得出该函数还是c接口想使用c语言来模拟面向对象的结构，其中的参数为：
				type：
				- CV_TERMCRIT_ITER  在当算法迭代次数超过max_iter的时候终止。
				- CV_TERMCRIT_EPS   在当算法得到的精度低于epsolon时终止；
				-CV_TERMCRIT_ITER+CV_TERMCRIT_EPS
				当算法迭代超过max_iter或者当获得的精度低于epsilon的时候，哪个先满足就停止
				max_iter：迭代的最大次数
				epsilon：要求的精度
				http://www.cnblogs.com/shouhuxianjian/p/4529174.html

				L231 trackWindow = selection;
				*/

				//trackWindow 为鼠标选择的区域，TermCriteria为确定迭代终止的准则
				RotatedRect trackBox = CamShift(backppoj, trackWindow, TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));

				if (trackWindow.area() <= 1)
				{
					int cols = backppoj.cols, rows = backppoj.rows, r = (MIN(cols, rows) + 5) / 6;
					trackWindow = Rect(trackWindow.x - r, trackWindow.y - r, trackWindow.x + r, trackWindow.y + r) & Rect(0, 0, cols, rows);


				}

				if (backprojMode)
				{
					cvtColor(backppoj, image, COLOR_GRAY2RGB);

				}
				/*
				void cvEllipse( CvArr* img, CvPoint center, CvSize axes, double angle,
				double start_angle, double end_angle, CvScalar color,
				int thickness=1, int line_type=8, int shift=0 );
				img                 图像。
				center              椭圆圆心坐标。

				axes                轴的长度。
				angle               偏转的角度。
				start_angle         圆弧起始角的角度。
				end_angle           圆弧终结角的角度。
				color               线条的颜色。
				thickness           线条的粗细程度。
				line_type           线条的类型,见CVLINE的描述。
				shift               圆心坐标点和数轴的精度。

				lineType – 线型
				Type of the line:
				8 (or omitted) - 8-connected line.
				4 - 4-connected line.
				CV_AA - antialiased line. 抗锯齿线。
				shift – 坐标点小数点位数.

				跟踪的时候以椭圆为代表目标
				*/

				ellipse(image, trackBox, Scalar(0, 0, 255), 3, CV_AA);
			}



		}

		//后面的代码是不管pause为真还是为假都要执行的 
		else if(trackObject<0)      //同时也是在按了暂停字母以后
		paused = false;

		if (selectObject && selection.width > 0 && selection.height > 0)
		{
			Mat roi(image, selection);
			bitwise_not(roi, roi);		//bitwise_not 为将每一个bit位取反


		}

		imshow("CamShift", image);
		imshow("直方图", histimg);
		int i;

		/*
	   waitKey(x);
	   第一个参数： 等待x ms，如果在此期间有按键按下，则立即结束并返回按下按键的
	   ASCII码，否则返回-1
	   如果x=0，那么无限等待下去，直到有按键按下
	   http://blog.sina.com.cn/s/blog_82a790120101jsp1.html
	   */

		char c = (char)waitKey(10);
		if (27 == c)     //退出键
			break;
		switch (c)
		{
		case 'b':		//反向投影模型 img/mask交替
			backprojMode = !backprojMode;
			break;
		case 'c':		//清零跟踪目标对象
				trackObject = 0;
				histimg = Scalar::all(0);
				break;
		case 'h':		//显示直方图交替
			showHist = !showHist;
			if (!showHist)
			{
				destroyWindow("直方图");
			}
			else
			{
				cvNamedWindow("直方图", 1);
			}
			break;

		case 'p':		//暂停跟踪交替
			paused = !paused;
			break;

		default:
			break;
		}


	}

	return 0;

}
