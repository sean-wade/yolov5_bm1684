/***
 * @Author: ZhangHao
 * @Date: 2020-12-14 10:35:50
 * @LastEditTime: 2020-12-14 10:35:51
 * @LastEditors: ZhangHao
 * @Description:
 * @FilePath: /yolov5_inference/yolov5.hpp
 * @zhanghao@yijiahe.com
 */

#ifndef __YOLOV5_HPP__
#define __YOLOV5_HPP__

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#define USE_OPENCV
#include "bm_wrapper.hpp"
#include "bmruntime_interface.h"
#include "zh_log.h"

using std::vector;

struct DetectRect {
  int left;
  int top;
  int right;
  int bottom;
  float score;
  int class_id;
};

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

inline static int nms_comparator(const DetectRect da, const DetectRect db)
{
    return da.score < db.score;
}

class YOLOV5
{
public:
    YOLOV5(const std::string bmodel);
    ~YOLOV5();
    void preForward(vector<cv::Mat>& images);
    void forward();
    vector<vector<DetectRect>> postForward();
    void drawResult();
    int getBatchSize();
    void writeBatchResultImg(std::string prefix);

private:
    void preprocess(bm_image& in, bm_image& out);
    void decodeResult(float* data, int yolo_idx, int frameWidth, int frameHeight);
    void nonMaxSuppression();
    void printDets();

private:
    /* handle of low level device */
    bm_handle_t bm_handle_;
    /* runtime helper */
    void *p_bmrt_;
    const char **net_names_;

    /* net infos */
    float threshold_prob_;
    float threshold_nms_;
    bool int8_flag_;
    int output_num_;

    /* net input */
    int net_h_;
    int net_w_;
    int batch_size_;
    int num_channels_;
    bm_shape_t input_shape_;

    // linear transformation arguments of BMCV
    bmcv_convert_to_attr convert_attr_;
    bm_image* scaled_inputs_;
    std::vector<cv::Mat> images_;

    /* net inference */
    int fm_size_[6];
    std::vector<int> output_sizes_;

    /* net outputs */
    std::vector<void*> outputs_;  //三个指针 -> float/int

    /* zhanghao */
    const size_t anchor_num_ = 3;
    const size_t classes_num_ = 2;
    vector<vector<vector<int>>> anchors_{{{10, 13}, {16, 30}, {33, 23}},
                                         {{30, 61}, {62, 45}, {59, 119}},
                                         {{116, 90}, {156, 198}, {373, 326}}};
    vector<DetectRect> dets_;
    vector<vector<DetectRect>> all_dets_;  //batch_size_

    /* draw */
    vector<cv::Scalar> colors_;
    vector<std::string> class_names_{"yw_gkxfw",
                                     "yw_nc"};
};



#endif