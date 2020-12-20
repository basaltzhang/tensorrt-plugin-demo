/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TRT_MASKRCNN_ROI_ALIGN_PLUGIN_H
#define TRT_MASKRCNN_ROI_ALIGN_PLUGIN_H
#include "maskrcnnRoIAlignPlugin/roiAlign.h"
#include "kernel.h"
#include "plugin.h"
#include <string>
#include <vector>
#include <iostream>

typedef unsigned short half_type;

using namespace nvinfer1::plugin;
namespace nvinfer1
{
namespace plugin
{

struct RoIAlignParameters
{
    int pooled_height;
    int pooled_width;
    float sampling_ratio;
    float spatial_scale;
};

class MaskrcnnRoIAlignPlugin: public IPluginV2DynamicExt
{
public:
    MaskrcnnRoIAlignPlugin(RoIAlignParameters param);

    MaskrcnnRoIAlignPlugin(const void* data, size_t length);

    ~MaskrcnnRoIAlignPlugin() override = default;

    int getNbOutputs() const override;

    //DynamicExt plugins returns DimsExprs class instead of Dims
    // Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
    DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) override;

    // DynamicExt plugin supportsFormat update.
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;

    int initialize() override;

    void terminate() override;

    // size_t getWorkspaceSize(int maxBatchSize) const override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;


    // int enqueue(
    //     int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, 
                const void* const* inputs, void* const* outputs, 
                void* workspace, 
                cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    // void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    //     const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    //     const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, 
             const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    IPluginV2DynamicExt* clone() const override;

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputType, int nbInputs) const override;

    void setPluginNamespace(const char* libNamespace) override;

    const char* getPluginNamespace() const override;

private:
    RoIAlignParameters param{};
    int numRoIs{};
    std::string mNamespace;
    const char* mPluginNamespace;
};

class MaskrcnnRoIAlignPluginCreator : public BaseCreator
{
public:
    MaskrcnnRoIAlignPluginCreator();

    ~MaskrcnnRoIAlignPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;
    RoIAlignParameters params;
    static std::vector<PluginField> mPluginAttributes;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_MASKRCNN_ROI_ALIGN_PLUGIN_H
