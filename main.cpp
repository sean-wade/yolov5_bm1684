/***
 * @Author: ZhangHao
 * @Date: 2020-12-14 11:02:17
 * @LastEditTime: 2020-12-14 11:02:17
 * @LastEditors: ZhangHao
 * @Description:
 * @FilePath: /yolov5_inference/main.cpp
 * @zhanghao@yijiahe.com
 */

#include <dirent.h>
#include "yolov5.h"

using namespace std;

void detect(YOLOV5 &net, vector<cv::Mat>& images)
{
    if(net.getBatchSize() != int(images.size()))
    {
        ERROR << "net size != images.size !!! ";
        return;
    }
    net.preForward(images);
    net.forward();
    vector<vector<DetectRect>> results = net.postForward();
}

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names)
{
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr)
    {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr)
    {
        if (strcmp(p_file->d_name, ".") != 0 && strcmp(p_file->d_name, "..") != 0)
        {
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

int main(int argc, char **argv)
{
    FNLog::FastStartDefaultLogger();
    INFO << "log init success";

    if (argc < 3)
    {
        ERROR << "USAGE:";
        WARN << "  " << argv[0] << " <image folder> <bmodel file> ";
        exit(1);
    }

    const char * image_folder = argv[1];
    std::vector<std::string> file_names;
    if (read_files_in_dir(image_folder, file_names) < 0)
    {
        ERROR << "read files in dir failed.";
        return -1;
    }

    string bmodel_file = argv[2];
    if (access(bmodel_file.c_str(), F_OK ) == -1)
    {
        ERROR << "Cannot find valid model file.";
        exit(1);
    }

    YOLOV5 net(bmodel_file);
    int batch_size = net.getBatchSize();

    for(int i=0; i < int(file_names.size()/batch_size); i++)
    {
        vector<cv::Mat> batch_imgs;
        for(int j=0; j<batch_size; j++)
        {
            cv::Mat img = cv::imread(string(image_folder) + "/" + file_names[i * batch_size + j], cv::IMREAD_COLOR, 0);
            batch_imgs.push_back(img);
        }
        detect(net, batch_imgs);
        net.writeBatchResultImg("res/" + to_string(i));
    }

}