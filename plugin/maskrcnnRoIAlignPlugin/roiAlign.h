/*************************************************************************
    > File Name: roiAlign.h
    > Author: Stewart
    > Mail: tanzhiyu@jd.com 
    > Created Time: Mon 23 Mar 2020 02:19:25 PM CST
 ************************************************************************/

#ifndef ROI_ALIGN_H
#define ROI_ALIGN_H

#include<iostream>
#include "plugin.h"
#include "cuda_runtime_api.h"
#include "kernel.h"

using namespace std;

pluginStatus_t RoIAlign(cudaStream_t stream,
    const int rois_per_image,
    const int channels,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const float* feature,
    const int feature_height, const int feature_width,
    const float spatial_scale,
    const float* rois,
    const int num_rois,
    float* output
    );

#endif
