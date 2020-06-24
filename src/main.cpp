
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// 图像混合
/*
void addWeightedDemo()
{
  cv::Mat src1 = cv::imread("/Users/zhangshijie/Desktop/dlrb.jpeg");
  Mat src2 = imread("/Users/zhangshijie/Desktop/jjj.jpeg");
  if (src1.empty() || src2.empty())
    return;
  cv::Mat out, multiplyOut;
  cv::cvtColor(src1, out, COLOR_RGB2GRAY);
  cv::imshow("Example1", src1);
  cv::imshow("Example2", src2);

  addWeighted(src1, 0.5, src2, 0.5, 0, out, -1);
  multiply(src1, src2, multiplyOut);
  cv::imshow("out", out);
  cv::imshow("multiplyOut", multiplyOut);
}
*/

// 对比度 demo
/*
void contrastDemo()
{

  Mat src1 = imread("/Users/zhangshijie/Desktop/dlrb.jpeg");
  Mat src2 = imread("/Users/zhangshijie/Desktop/jjj.jpeg");
  if (src1.empty() || src2.empty())
    return;

  Mat dst;
  char input_win[] = "input image";

  float alpha = 0.5;
  float beta = 0;

  // cvtColor(src1, src1, COLOR_BGR2GRAY);

  dst = Mat::zeros(src1.size(), src1.type());
  int height = src1.rows;
  int width = src1.cols;

  imshow(input_win, src1);

  for (size_t row = 0; row < height; row++)
  {
    for (size_t col = 0; col < width; col++)
    {
      if (src1.channels() == 3)
      {
        Vec3b pix = src1.at<Vec3b>(row, col);
        int b = pix[0];
        int g = pix[1];
        int r = pix[2];
        dst.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(b * alpha + beta);
        dst.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(g * alpha + beta);
        dst.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(r * alpha + beta);
      }
      else if (src1.channels() == 1)
      {
        int v = src1.at<uchar>(row, col);
        dst.at<uchar>(row, col) = saturate_cast<uchar>(v * alpha + beta);
      }
    }
  }

  imshow("out", dst);
}
*/

// 图形绘制
/* 
void myLiners(Mat bgImage)
{

  Point p1 = Point(20, 30);
  Point p2;
  p2.x = 100;
  p2.y = 100;

  Scalar color = Scalar(0, 0, 255);

  line(bgImage, p1, p2, color, 1, LINE_8);
}

void myEllipse(Mat bgImage)
{
  Scalar color = Scalar(0, 255, 0);
  ellipse(bgImage, Point(bgImage.cols / 2, bgImage.rows / 2), Size(100, 50), 360, 0, 360, color, 2, LINE_8);
}

void myCircle(Mat bgImage)
{
  Scalar color = Scalar(0, 255, 255);
  circle(bgImage, Point(bgImage.cols / 2, bgImage.rows / 2), 50, color);
}

void myPolygon(Mat bgImage)
{

  Point pts[1][5];

  pts[0][0] = Point(10, 10);
  pts[0][1] = Point(10, 20);
  pts[0][2] = Point(20, 20);
  pts[0][3] = Point(20, 10);
  pts[0][4] = Point(10, 10);

  const Point *ppts[] = {pts[0]};
  int npt[] = {5};
  Scalar color = Scalar(0, 255, 255);

  fillPoly(bgImage, ppts, npt, 1, color, 8);
}

void myText(Mat bgImage)
{
  Scalar color = Scalar(0, 255, 255);
  putText(bgImage, "Hello World", Point(100, 100), FONT_HERSHEY_COMPLEX, 1, color);
}

void myRandomLineDemo(Mat bgImage)
{

  Mat newBg = Mat::zeros(bgImage.size(), bgImage.type());
  RNG rng(12345);

  for (int i = 0; i < 1000; i++)
  {
    Point pt1, pt2;
    pt1.x = rng.uniform(0, bgImage.cols);
    pt1.y = rng.uniform(0, bgImage.rows);

    pt2.x = rng.uniform(0, bgImage.cols);
    pt2.y = rng.uniform(0, bgImage.rows);

    Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    line(newBg, pt1, pt2, color, 1, 8);
    if (waitKey(50) > 0)
    {
      break;
    }
    imshow("randow line window", newBg);
  }
}

void drawDemo()
{

  Mat src = imread("/Users/zhangshijie/Desktop/dlrb.jpeg");
  myLiners(src);
  myEllipse(src);
  myCircle(src);
  myPolygon(src);
  myText(src);
  myRandomLineDemo(src);
  imshow("src", src);
}
*/

/*
// 均值滤波
void blurDemo()
{
  Mat src = imread("/Users/zhangshijie/Desktop/dlrb.jpeg");
  Mat dst;

  blur(src, dst, Size(15, 1), Point(-1, -1));

  imshow("src", src);
  imshow("blur", dst);
}

// 高斯滤波
void GaussianBlurDemo()
{
  Mat src = imread("/Users/zhangshijie/Desktop/dlrb.jpeg");
  Mat dst;

  GaussianBlur(src, dst, Size(7, 7), 1, 1);

  imshow("src", src);
  imshow("GaussianBlur", dst);
}

// 中值模糊
void medianBlurDemo()
{
  Mat src = imread("/Users/zhangshijie/Desktop/dlrb.jpeg");
  Mat dst;
  medianBlur(src, dst, 5);
  imshow("src", src);
  imshow("medianBlurDemo", dst);
}

*/

// 双边高斯滤波
/*
void bilateralFilterDemo() {

  Mat src = imread("/Users/zhangshijie/Desktop/dlrb.jpeg");
  Mat dst;
  bilateralFilter(src, dst, 7, 50, 5);

  Mat resultImg;
  Mat kernel = (Mat_<int>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
  filter2D(dst, resultImg, -1, kernel, Point(-1, -1), 0);

  imshow("src", src);
  imshow("bilateralFilterDemo", dst); 
  imshow("resultImg", resultImg);
}
*/
/*
int element_size = 3;
int max_size = 21;
Mat src, dst;

void CallBack_Demo(int, void *);

void CallBack_Demo(int, void *)
{
  int s = element_size * 2 + 1;
  Mat structureElement = getStructuringElement(MORPH_RECT, Size(s, s), Point(-1, -1));
  // 膨胀
  // dilate(src, dst, structureElement, Point(-1, -1));
  // 腐蚀
  erode(src, dst, structureElement);
  imshow("dst", dst);
}

void dilateDemo()
{
  src = imread("/Users/zhangshijie/Desktop/dlrb.jpeg");
  imshow("src", src);
  namedWindow("dst", WINDOW_AUTOSIZE);
  createTrackbar("ELement", "dst", &element_size, max_size, CallBack_Demo);
  CallBack_Demo(0, 0);

}
*/

/*
// 开操作
// 先腐蚀后膨胀， 去除黑点
void openDemo()
{
  Mat src = imread("/Users/zhangshijie/Desktop/dlrb.jpeg");
  Mat dst;

  imshow("src", src);

  Mat kernel = getStructuringElement(MORPH_RECT, Size(11, 11), Point(-1, -1));
  morphologyEx(src, dst, MORPH_OPEN, kernel);

  imshow("dst", dst);
}

// 关操作，先膨胀后腐蚀, 去除白点
void closeDemo()
{
  Mat src = imread("/Users/zhangshijie/Desktop/dlrb.jpeg");
  Mat dst;

  imshow("src", src);

  Mat kernel = getStructuringElement(MORPH_RECT, Size(11, 11), Point(-1, -1));
  morphologyEx(src, dst, MORPH_CLOSE, kernel);

  imshow("dst", dst);
}

// 梯度操作，膨胀减去腐蚀
void gradientDemo()
{

  Mat src = imread("/Users/zhangshijie/Desktop/dlrb.jpeg");
  Mat dst;

  imshow("src", src);

  Mat kernel = getStructuringElement(MORPH_RECT, Size(11, 11), Point(-1, -1));
  morphologyEx(src, dst, MORPH_GRADIENT, kernel);

  imshow("dst", dst);
}

// 顶帽，原图片和开操作之后的差值
void topHat()
{
  Mat src = imread("/Users/zhangshijie/Desktop/dlrb.jpeg");
  Mat dst;

  imshow("src", src);

  Mat kernel = getStructuringElement(MORPH_RECT, Size(11, 11), Point(-1, -1));
  morphologyEx(src, dst, MORPH_TOPHAT, kernel);

  imshow("dst", dst);
}

// 黑帽，闭操作和原图之间的差值
void blackHat() {
   Mat src = imread("/Users/zhangshijie/Desktop/dlrb.jpeg");
  Mat dst;

  imshow("src", src);

  Mat kernel = getStructuringElement(MORPH_RECT, Size(11, 11), Point(-1, -1));
  morphologyEx(src, dst, MORPH_BLACKHAT, kernel);

  imshow("dst", dst);
}
*/

/*
void horizontalLineDemo()
{
  Mat src = imread("/Users/zhangshijie/Desktop/dlrb.jpeg");
  imshow("src", src);

  Mat grey_src;
  cvtColor(src, grey_src, COLOR_BGR2GRAY);
  imshow("grey_src", grey_src);

  Mat binImg;
  adaptiveThreshold(~grey_src, binImg, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
  imshow("binImg", binImg);

  Mat hline = getStructuringElement(MORPH_RECT, Size(src.cols / 16, 1), Point(-1, -1));
  Mat vline = getStructuringElement(MORPH_RECT, Size(1, src.rows / 16), Point(-1, -1));

  Mat dst;
  // erode(binImg, dst, vline);
  // dilate(dst, dst, vline);
  morphologyEx(binImg, dst, MORPH_OPEN, hline);
  bitwise_not(dst, dst);
  blur(dst, dst, Size(3, 3));
  imshow("dst", dst);
}
*/

/*
void pryDownAndUpDemo()
{
  Mat src = imread("/Users/zhangshijie/Desktop/dlrb.jpeg");
  imshow("src", src);

  Mat dst;
  // 上采样
  // pyrUp(src, dst, Size(src.cols * 2, src.rows * 2));
  pyrDown(src, dst, Size(src.cols / 2, src.rows / 2));

  imshow("dst", dst);

  Mat g1, g2, gray_src, dogImg;
  // 高斯不同
  cvtColor(src, gray_src, COLOR_BGR2GRAY);
  GaussianBlur(gray_src, g1, Size(5, 5), 0, 0);
  GaussianBlur(g1, g2, Size(5, 5), 0, 0);

  subtract(g1, g2, dogImg, Mat());
  //归一化显示
  normalize(dogImg, dogImg, 255, 0, NORM_MINMAX);
  imshow("dogImg", dogImg);
}
*/

// 阀值
/*
Mat src, grey_src, dst;
int threshold_value = 127;
int threshold_max = 255;
int type_value = 2;
int type_max = 4;

void thresholdCallback(int , void*) {
  cvtColor(src, grey_src, COLOR_BGR2GRAY);
  // 二值化
  // threshold(grey_src, dst, threshold_value, threshold_max, THRESH_BINARY);
  // 反向二值化
  threshold(grey_src, dst, threshold_value, threshold_max, THRESH_TRIANGLE | type_value);
  imshow("dst", dst);
}

void therodholdDemo()
{
  src = imread("/Users/zhangshijie/Desktop/dlrb.jpeg");
  imshow("src", src);

  cvtColor(src, grey_src, COLOR_BGR2GRAY);
  namedWindow("dst");
  createTrackbar("dst", "dst", &threshold_value, threshold_max, thresholdCallback);
  createTrackbar("dst_1", "dst", &type_value, type_max, thresholdCallback);
  thresholdCallback(0, 0);
}
*/

// 卷积
void convolutionDemo()
{

  Mat src = imread("/Users/zhangshijie/Desktop/dlrb.jpeg");
  imshow("src", src);

  Mat dst;
  // Robert X 方向
  // 1, 0
  // 0, -1
  Mat kernel = (Mat_<int>(2, 2) << 1, 0, 0, -1);

  filter2D(src, dst, -1, kernel, Point(-1, -1));
  imshow("Robert X", dst);

  // Robert Y 方向
  // 0,1
  // -1, 0
  Mat yKernel = (Mat_<int>(2, 2) << 0, 1, -1, 0);
  filter2D(src, dst, -1, yKernel, Point(-1, -1));
  imshow("Robert Y", dst);

  // Sobel X 算子
  Mat sX = (Mat_<int>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
  filter2D(src, dst, -1, sX, Point(-1, -1));
  imshow("Sobel X", dst);

  // Sobel Y 算子
  Mat sY = (Mat_<int>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
  filter2D(src, dst, -1, sY, Point(-1, -1));
  imshow("Sobel Y", dst);

  // 拉普拉斯算子
  Mat lplsY = (Mat_<int>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
  filter2D(src, dst, -1, lplsY, Point(-1, -1));
  imshow("拉普拉斯算子", dst);

  int c = 0;
  int index = 0;
  // 模糊算法
  while (true)
  {
    c = waitKey(500);
    if ((char)c == 27)
    { // ESC
      break;
    }
    int ksize = 4 + (index % 5) * 2 + 1;
    Mat kernel = Mat::ones(Size(ksize, ksize), CV_32F) / (float)(ksize * ksize);

    filter2D(src, dst, -1, kernel, Point(-1, -1));
    imshow("模糊算法", dst);
    index++;
  }
}

int main(int argc, char **argv)
{

  // addWeightedDemo();
  // contrastDemo();
  // drawDemo();
  // blurDemo();
  // GaussianBlurDemo();
  // medianBlurDemo();
  // bilateralFilterDemo();
  // dilateDemo();

  // openDemo();
  // closeDemo();
  // gradientDemo();
  // topHat();
  // blackHat();

  // horizontalLineDemo();

  // pryDownAndUpDemo();

  // therodholdDemo();

  convolutionDemo();

  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}