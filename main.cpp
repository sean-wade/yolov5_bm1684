/***
 * @Author: ZhangHao
 * @Date: 2020-12-14 11:02:17
 * @LastEditTime: 2020-12-14 11:02:17
 * @LastEditors: ZhangHao
 * @Description:
 * @FilePath: /yolov5_inference/main.cpp
 * @zhanghao@yijiahe.com
 */

#include <boost/filesystem.hpp>
#include "yolov5.h"

namespace fs = boost::filesystem;
using namespace std;

int main(int argc, char **argv)
{
    cout.setf(ios::fixed);
    if (argc < 4)
    {
        cout << "USAGE:" << endl;
        cout << "  " << argv[0] << " image <image list> <bmodel file> " << endl;
        cout << "  " << argv[0] << " video <video url>  <bmodel file> " << endl;
        exit(1);
    }

    bool is_video = false;
    if (strcmp(argv[1], "video") == 0)
    {
        is_video = true;
    }
    else if (strcmp(argv[1], "image") == 0)
    {
        is_video = false;
    }
    else
    {
        cout << "Wrong input type, neither image nor video." << endl;
        exit(1);
    }

    string image_list = argv[2];
    if (!is_video && !fs::exists(image_list))
    {
        cout << "Cannot find input image file." << endl;
        exit(1);
    }

    string bmodel_file = argv[3];
    if (!fs::exists(bmodel_file))
    {
        cout << "Cannot find valid model file." << endl;
        exit(1);
    }

    YOLOV5 net(bmodel_file);

    for(int j=0; j<1000; j++)
    {
        cv::Mat img = cv::imread("test.jpg", cv::IMREAD_COLOR, 0);
        vector<cv::Mat> batch_imgs;
        batch_imgs.push_back(img);

        net.preForward(batch_imgs);
        net.forward();
        net.postForward();
    }

}