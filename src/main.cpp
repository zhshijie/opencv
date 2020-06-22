
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
  cv::Mat img = cv::imread("/Users/zhangshijie/Desktop/WechatIMG3801.jpeg", -1);
  if (img.empty())
    return -1;
  cv::Mat out;
  cv::cvtColor(img, out, COLOR_RGB2GRAY);
  cv::imshow("Example1", img);
  cv::imshow("Example2", out);
  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}