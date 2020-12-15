/***
 * @Author: ZhangHao
 * @Date: 2020-12-14 10:41:14
 * @LastEditTime: 2020-12-14 10:41:14
 * @LastEditors: ZhangHao
 * @Description:
 * @FilePath: /yolov5_inference/yolov5.cpp
 * @zhanghao@yijiahe.com
 */

#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include "yolov5.h"

using std::vector;
using std::max;
using std::min;

YOLOV5::YOLOV5(const std::string bmodel)
{
    /* create device handler */
    bm_dev_request(&bm_handle_, 0);
    /* create inference runtime handler */
    p_bmrt_ = bmrt_create(bm_handle_);
    /* load bmodel by file */
    bool flag = bmrt_load_bmodel(p_bmrt_, bmodel.c_str());
    if (!flag)
    {
        std::cout << "ERROR: failed to load bmodel[" << bmodel << "] " << std::endl;
        exit(-1);
    }
    bmrt_get_network_names(p_bmrt_, &net_names_);
    std::cout << "> Load model " << net_names_[0] << " successfully" << std::endl;

    /* more info pelase refer to bm_net_info_t in bmdef.h */
    auto net_info = bmrt_get_network_info(p_bmrt_, net_names_[0]);
    std::cout << "> input scale:"   << net_info->input_scales[0]  << std::endl;
    std::cout << "> output scale:"  << net_info->output_scales[0] << std::endl;
    std::cout << "> input number:"  << net_info->input_num        << std::endl;
    std::cout << "> output number:" << net_info->output_num       << std::endl;
    bm_image_data_format_ext data_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;

    /* TODO: get class number from net_info */

    /* get fp32/int8 type, the thresholds may be different */
    if (BM_FLOAT32 == net_info->input_dtypes[0])
    {
        threshold_prob_ = 0.5;
        threshold_nms_ = 0.45;
        int8_flag_ = false;
        std::cout << "> input_dtypes is fp32 model" << std::endl;
        data_type = DATA_TYPE_EXT_FLOAT32;
    }
    else
    {
        threshold_prob_ = 0.2;
        threshold_nms_ = 0.45;
        int8_flag_ = true;
        std::cout << "> input_dtypes is int8 model" << std::endl;
    }
    bmrt_print_network_info(net_info);

    /*
    * only one input shape supported in the pre-built model
    * you can get stage_num from net_info
    */
    auto &input_shape = net_info->stages[0].input_shapes[0];
    /* malloc input and output system memory for preprocess data */
    int count = bmrt_shape_count(&input_shape);
    std::cout << "> input count:" << count << std::endl;

    output_num_ = net_info->output_num;
    for (int i = 0; i < output_num_; i++)
    {
        auto &output_shape = net_info->stages[0].output_shapes[i];
        count = bmrt_shape_count(&output_shape);
        std::cout << "> output " << i << " count:" << count << std::endl;

        float* out = new float[count];
        outputs_.push_back(out);
        fm_size_[i * 2] = output_shape.dims[3];
        fm_size_[i * 2 + 1] = output_shape.dims[2];
        output_sizes_.push_back(output_shape.dims[1] *
                                output_shape.dims[2] *
                                output_shape.dims[3] *
                                output_shape.dims[4]);
    }

    batch_size_ = input_shape.dims[0];
    num_channels_ = input_shape.dims[1];
    net_h_ = input_shape.dims[2];
    net_w_ = input_shape.dims[3];
    input_shape_ = {4, {batch_size_, 3, net_h_, net_w_}};

    float input_scale = 1.0 / 255;
    if (int8_flag_)
    {
        input_scale *= net_info->input_scales[0];
    }
    convert_attr_.alpha_0 = input_scale;
    convert_attr_.beta_0 = 0;
    convert_attr_.alpha_1 = input_scale;
    convert_attr_.beta_1 = 0;
    convert_attr_.alpha_2 = input_scale;
    convert_attr_.beta_2 = 0;
    scaled_inputs_ = new bm_image[batch_size_];
    /* create bm_image - used for border processing */
    bm_status_t ret = bm_image_create_batch(bm_handle_, net_h_, net_w_,
                        FORMAT_RGB_PLANAR,
                        data_type,
                        scaled_inputs_, batch_size_);
    if (BM_SUCCESS != ret)
    {
        std::cerr << "ERROR: bm_image_create_batch failed" << std::endl;
        exit(1);
    }

    // for draw
    cv::RNG rng(cv::getTickCount());
    for(size_t j=0; j<classes_num_; j++)
    {
        colors_.push_back(cv::Scalar(rng.uniform(0,255),
                                     rng.uniform(0,255),
                                     rng.uniform(0,255)));
    }
}

YOLOV5::~YOLOV5()
{
    bm_image_destroy_batch(scaled_inputs_, batch_size_);
    if (scaled_inputs_)
    {
        delete []scaled_inputs_;
    }
    for (size_t i = 0; i < outputs_.size(); i++)
    {
        delete [] reinterpret_cast<float*>(outputs_[i]);
    }
    free(net_names_);
    bmrt_destroy(p_bmrt_);
    bm_dev_free(bm_handle_);
}

void YOLOV5::preprocess(bm_image& in, bm_image& out)
{
    bm_image_create(bm_handle_, net_h_, net_w_, FORMAT_RGB_PLANAR,
                    DATA_TYPE_EXT_1N_BYTE, &out, NULL);
    bmcv_rect_t crop_rect = {0, 0, in.width, in.height};
    bmcv_image_vpp_convert(bm_handle_, 1, in, &out, &crop_rect);
}

void YOLOV5::preForward(std::vector<cv::Mat>& images)
{
    images_.clear();
    all_dets_.clear();
    vector<bm_image> processed_imgs;
    for (size_t i = 0; i < images.size(); i++)
    {
        bm_image bmimg;
        bm_image processed_img;
        bm_image_from_mat(bm_handle_, images[i], bmimg);
        preprocess(bmimg, processed_img);
        bm_image_destroy(bmimg);
        processed_imgs.push_back(processed_img);
        images_.push_back(images[i]);
    }
    bmcv_image_convert_to(bm_handle_,
                          batch_size_,
                          convert_attr_,
                          &processed_imgs[0],
                          scaled_inputs_);

    for (size_t i = 0; i < images.size(); i++)
    {
        bm_image_destroy(processed_imgs[i]);
    }
}

void YOLOV5::forward()
{
    bool res = bm_inference(p_bmrt_,
                            scaled_inputs_,
                            outputs_,
                            input_shape_,
                            reinterpret_cast<const char*>(net_names_[0]));
    if (!res)
    {
        std::cout << "ERROR : inference failed!!"<< std::endl;
        exit(1);
    }
}

vector<vector<DetectRect>> YOLOV5::postForward()
{
    for (int i = 0; i < batch_size_; i++)
    {
        dets_.clear();
        for (int j = 0; j < output_num_; j++)
        {
            float* blob = reinterpret_cast<float*>(outputs_[j]) + output_sizes_[j] * i;
            decodeResult(blob, j, images_[i].cols, images_[i].rows);
        }
        nonMaxSuppression();
        all_dets_.push_back(dets_);
    }
    printDets();
    drawResult();
    return all_dets_;
}


void YOLOV5::decodeResult(float* data, int yolo_idx, int frameWidth, int frameHeight)
{
    // float* data: 134400(1*3*80*80*7) 或 33600(1*3*40*40*7) 或 8400(1*3*20*20*7)
    int stride_w = int(net_w_ / fm_size_[2 * yolo_idx + 1]);   //下采样倍率，注意这里的w/h可能是反的
    int stride_h = int(net_h_ / fm_size_[2 * yolo_idx]);
    int fm_area_ = fm_size_[2 * yolo_idx] * fm_size_[2 * yolo_idx + 1];
    // std::cout << "fm_area_ = "<<fm_area_<<std::endl;
    for(size_t c = 0; c < anchor_num_; c++)
    {
        const float* ptr = data + output_sizes_[yolo_idx] / anchor_num_ * c;
        for(int y = 0; y < fm_area_; y++)  // 遍历 fm (80*80)
        {
            float score = sigmoid(ptr[4]);
            if(score > 0.3)
            {
                // std::cout << "score : " << score << std::endl;
                // vector<float> det(6);
                DetectRect det;
                // (int)(net_w_ / stride_w)) 可以用 fm_size_ 代替
                float centerX = (sigmoid(ptr[0]) * 2 - 0.5 + y % (int)(net_w_ / stride_w)) * stride_w * frameWidth / net_w_; //center_x
                float centerY = (sigmoid(ptr[1]) * 2 - 0.5 + (int)(y / (net_h_ / stride_h))) * stride_h * frameHeight / net_h_; //center_y
                float width   = pow((sigmoid(ptr[2]) * 2), 2) * anchors_[yolo_idx][c][0] * frameWidth  / net_w_; //w
                float height  = pow((sigmoid(ptr[3]) * 2), 2) * anchors_[yolo_idx][c][1] * frameHeight / net_h_; //h

                det.left   = int(centerX - width  / 2);
                det.top    = int(centerY - height / 2);
                det.right  = int(centerX + width  / 2);
                det.bottom = int(centerY + height / 2);
                det.score  = 0;

                for (size_t i=5; i < classes_num_ + 5; i++)
                {
                    float conf = sigmoid(ptr[i]);
                    if (conf * score > det.score)
                    {
                        det.score    = conf * score;
                        det.class_id = i - 5;
                    }
                }
                dets_.push_back(det);
            }
            ptr += (classes_num_ + 5);
        }
    }
}

void YOLOV5::nonMaxSuppression()
{
    int length = dets_.size();
    int index = length - 1;

    sort(dets_.begin(), dets_.end(), nms_comparator);
    vector<float> areas(length);
    for (int i=0; i<length; i++)
    {
        areas[i] = (dets_[i].bottom - dets_[i].top) * (dets_[i].right - dets_[i].left);
    }

    while (index  > 0)
    {
        int i = 0;
        while (i < index)
        {
            float left    = max(dets_[index].left,   dets_[i].left);
            float top     = max(dets_[index].top,    dets_[i].top);
            float right   = min(dets_[index].right,  dets_[i].right);
            float bottom  = min(dets_[index].bottom, dets_[i].bottom);
            float overlap = max(0.0f, right - left) * max(0.0f, bottom - top);
            if (overlap / (areas[index] + areas[i] - overlap) > threshold_nms_)
            {
                areas.erase(areas.begin() + i);
                dets_.erase(dets_.begin() + i);
                index --;
            }
            else
            {
                i++;
            }
        }
        index--;
    }
}

void YOLOV5::printDets()
{
    std::cout << " >>>> Detection results: " << std::endl;
    for(size_t j=0; j < all_dets_.size(); j++)
    {
        std::cout << " >>>> The " << j + 1 << "th image: " << std::endl;
        vector<DetectRect> det = all_dets_[j];
        for(size_t i=0; i<det.size(); i++)
        {
            std::cout << " class = "  << det[i].class_id
                      << " score = "  << det[i].score
                      << " left = "   << det[i].left
                      << " top = "    << det[i].top
                      << " right = "  << det[i].right
                      << " bottom = " << det[i].bottom
                      << std::endl;
        }
    }

}

void YOLOV5::drawResult()
{
    if(all_dets_.size() != images_.size())
    {
        std::cout << "result.size != images.size !!! " << std::endl;
        return;
    }
    for(size_t j = 0; j < all_dets_.size(); j++)
    {
        vector<DetectRect> det = all_dets_[j];
        int thick = int(0.002 * (images_[j].cols + images_[j].rows) / 2) + 2;  // line/font thickness
        for (size_t i=0; i < det.size(); i++)
        {
            cv::rectangle(images_[j],
                          cv::Rect(det[i].left, det[i].top, (det[i].right - det[i].left), (det[i].bottom - det[i].top)),
                          colors_[det[i].class_id],
                          thick);

            int tf = max(thick - 1, 1);
            cv::String label = cv::format("%s:%.2f", class_names_[det[i].class_id].c_str(), det[i].score);
            cv::Size t_size = cv::getTextSize(label, 0, thick / 3, tf, 0);
            cv::rectangle(images_[j],
                          cv::Point(det[i].left, det[i].top), cv::Point(det[i].left + t_size.width, det[i].top - t_size.height - 3),
                          colors_[det[i].class_id],
                          -1,
                          cv::LINE_AA);  // filled
            cv::putText(images_[j],
                        label,
                        cv::Point(det[i].left, det[i].top - 2),
                        cv::FONT_HERSHEY_SIMPLEX,
                        thick / 3,
                        cv::Scalar(255, 255, 255),
                        tf,
                        cv::LINE_AA);
        }
        cv::imwrite(cv::format("res_%ld.jpg", j), images_[j]);
    }
}
