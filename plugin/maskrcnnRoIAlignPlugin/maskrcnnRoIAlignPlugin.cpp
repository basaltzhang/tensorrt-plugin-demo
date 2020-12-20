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

#include "maskrcnnRoIAlignPlugin.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::MaskrcnnRoIAlignPlugin;
using nvinfer1::plugin::MaskrcnnRoIAlignPluginCreator;
using nvinfer1::plugin::RoIAlignParameters;

namespace
{
constexpr const char* ROI_ALIGN_PLUGIN_VERSION{"1"};
constexpr const char* ROI_ALIGN_PLUGIN_NAME{"MaskrcnnRoIAlign_TRT"};
} // namespace
REGISTER_TENSORRT_PLUGIN(MaskrcnnRoIAlignPluginCreator);

PluginFieldCollection MaskrcnnRoIAlignPluginCreator::mFC{};
std::vector<PluginField> MaskrcnnRoIAlignPluginCreator::mPluginAttributes;

MaskrcnnRoIAlignPlugin::MaskrcnnRoIAlignPlugin(RoIAlignParameters params)
    : param(params)
{
}

MaskrcnnRoIAlignPlugin::MaskrcnnRoIAlignPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    param = read<RoIAlignParameters>(d);
    numRoIs = read<int>(d);
    ASSERT(d == a + length);
}

int MaskrcnnRoIAlignPlugin::getNbOutputs() const
{
    return 1;
}

int MaskrcnnRoIAlignPlugin::initialize()
{
    return STATUS_SUCCESS;
}

void MaskrcnnRoIAlignPlugin::terminate() {}

DimsExprs MaskrcnnRoIAlignPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    ASSERT(nbInputs == 2);
    ASSERT(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    ASSERT(inputs[0].nbDims == 4);
    ASSERT(inputs[1].nbDims == 2);

    // num_detections
    DimsExprs ret;
    ret.nbDims = 4;
    ret.d[0] = inputs[1].d[0];
    ret.d[1] = inputs[0].d[1];
    ret.d[2] = exprBuilder.constant(param.pooled_height);
    ret.d[3] = exprBuilder.constant(param.pooled_width);
    return ret;
}

size_t MaskrcnnRoIAlignPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const
{
    return 0;
}

int MaskrcnnRoIAlignPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs,
                void* workspace,
                cudaStream_t stream)
{
    const float* const feature = (const float* const)(inputs[0]);
    const float* const rois = (const float* const)(inputs[1]);

    float* output = (float*)(outputs[0]);
    pluginStatus_t status = RoIAlign(
            stream,
            inputDesc[0].dims.d[0],
            inputDesc[0].dims.d[1],
            param.pooled_height,
            param.pooled_width,
            param.sampling_ratio,
            feature,
            inputDesc[0].dims.d[2], inputDesc[0].dims.d[3],
            param.spatial_scale,
            rois, inputDesc[1].dims.d[0],
            output
            );
    // float* a = (float*)malloc(200 * sizeof(float));
    // cudaMemcpy(a, output, 200 * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int b = 0; b < 100; b ++) {
    //     std::cout << ", " << a[b];
    // }
    // std::cout  << std::endl;
    // free(a);
    // cudaStreamSynchronize(stream);
    // std::cout << "outputDesc[0].dims.d[0]: " << outputDesc[0].dims.d[0] << ", outputDesc[0].dims.d[1]: " << outputDesc[0].dims.d[1] << ", outputDesc[0].dims.d[2]: " << outputDesc[0].dims.d[2] << ", outputDesc[0].dims.d[3]: " << outputDesc[0].dims.d[3] << std::endl;
    // float* a = (float*)malloc(800 * 112 * 49 * sizeof(float));
    // cudaMemcpy(a, output, 800 * 112 * 49 * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 800; i += 400) {
    //     for (int b = 0; b < 1; b ++) {
    //         for (int j = 0; j < 1; j ++) {
    //             for (int k = 0; k < 5; k ++) {
    //                 std::cout << a[i * 112 * 7 * 7 + b * 7 * 7 + j * 7 + k] << ", ";
    //             }
    //         }
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout  << std::endl;
    // free(a);

    ASSERT(status == STATUS_SUCCESS);
    return 0;
}

size_t MaskrcnnRoIAlignPlugin::getSerializationSize() const
{
    // RoIAlignParameters, numRoIs
    return sizeof(RoIAlignParameters) + sizeof(int);
}

void MaskrcnnRoIAlignPlugin::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, param);
    write(d, numRoIs);
    ASSERT(d == a + getSerializationSize());
}

void MaskrcnnRoIAlignPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
             const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    ASSERT(nbInputs == 2);
    ASSERT(nbOutputs == 1);
    ASSERT(in[0].desc.dims.nbDims == 4);
    ASSERT(in[1].desc.dims.nbDims == 2);
    numRoIs = in[1].desc.dims.d[0];
    ASSERT(in[1].desc.dims.d[1] == 5 || in[1].desc.dims.d[1] == -1);
}


bool MaskrcnnRoIAlignPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF)
        && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR && inOut[pos].type == inOut[0].type);
}

const char* MaskrcnnRoIAlignPlugin::getPluginType() const
{
    return ROI_ALIGN_PLUGIN_NAME;
}

const char* MaskrcnnRoIAlignPlugin::getPluginVersion() const
{
    return ROI_ALIGN_PLUGIN_VERSION;
}

void MaskrcnnRoIAlignPlugin::destroy()
{
    delete this;
}

IPluginV2DynamicExt* MaskrcnnRoIAlignPlugin::clone() const
{
    auto* plugin = new MaskrcnnRoIAlignPlugin(param);
    plugin->numRoIs = numRoIs;
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void MaskrcnnRoIAlignPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* MaskrcnnRoIAlignPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

nvinfer1::DataType MaskrcnnRoIAlignPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    ASSERT(index >= 0 && index < this->getNbOutputs());
    return nvinfer1::DataType::kFLOAT;
}

MaskrcnnRoIAlignPluginCreator::MaskrcnnRoIAlignPluginCreator()
    : params{}
{
    // Plugin field meta data {name,  data, type, length}
    mPluginAttributes.emplace_back(PluginField("pooled_height", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("pooled_width", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("sampling_ratio", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("spatial_scale", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* MaskrcnnRoIAlignPluginCreator::getPluginName() const
{
    return ROI_ALIGN_PLUGIN_NAME;
}

const char* MaskrcnnRoIAlignPluginCreator::getPluginVersion() const
{
    return ROI_ALIGN_PLUGIN_VERSION;
}

const PluginFieldCollection* MaskrcnnRoIAlignPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2DynamicExt* MaskrcnnRoIAlignPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;

    for (int i = 0; i < fc->nbFields; ++i) {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "pooled_height")) {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.pooled_height = *(static_cast<const int*>(fields[i].data));
        } else if (!strcmp(attrName, "pooled_width")) {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.pooled_width = *(static_cast<const int*>(fields[i].data));
        } else if (!strcmp(attrName, "sampling_ratio")) {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.sampling_ratio = *(static_cast<const float*>(fields[i].data));
        } else if (!strcmp(attrName, "spatial_scale")) {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.spatial_scale = *(static_cast<const float*>(fields[i].data));
        // } else if (!strcmp(attrName, "spatial_scales")) {
        //     ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
        //     const int size = fields[i].length;
        //     const float* o = static_cast<const float*>(fields[i].data);
        //     for (int j = 0; j < size; j++)
        //     {
        //         params.spatial_scales[j] = *o;
        //         o++;
        //     }
        }
    }

    MaskrcnnRoIAlignPlugin* plugin = new MaskrcnnRoIAlignPlugin(params);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2DynamicExt* MaskrcnnRoIAlignPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call NMS::destroy()
    MaskrcnnRoIAlignPlugin* plugin = new MaskrcnnRoIAlignPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
