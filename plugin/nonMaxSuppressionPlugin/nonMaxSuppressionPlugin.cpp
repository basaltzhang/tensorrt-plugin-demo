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

#include "nonMaxSuppressionPlugin.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::NonMaxSuppressionPlugin;
using nvinfer1::plugin::NonMaxSuppressionPluginCreator;
using nvinfer1::plugin::NMSParameters;

namespace
{
constexpr const char* NMS_PLUGIN_VERSION{"1"};
constexpr const char* NMS_PLUGIN_NAME{"NonMaxSuppression_TRT"};
} // namespace
REGISTER_TENSORRT_PLUGIN(NonMaxSuppressionPluginCreator);

PluginFieldCollection NonMaxSuppressionPluginCreator::mFC{};
std::vector<PluginField> NonMaxSuppressionPluginCreator::mPluginAttributes;

NonMaxSuppressionPlugin::NonMaxSuppressionPlugin(NMSParameters params)
    : param(params)
{
}

NonMaxSuppressionPlugin::NonMaxSuppressionPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    param = read<NMSParameters>(d);
    boxesSize = read<int>(d);
    scoresSize = read<int>(d);
    numPriors = read<int>(d);
    mClipBoxes = read<bool>(d);
    ASSERT(d == a + length);
}

int NonMaxSuppressionPlugin::getNbOutputs() const
{
    return 1;
}

int NonMaxSuppressionPlugin::initialize()
{
    return STATUS_SUCCESS;
}

void NonMaxSuppressionPlugin::terminate() {}

// Dims BatchedNMSPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
DimsExprs NonMaxSuppressionPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    ASSERT(nbInputs == 2);
    ASSERT(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    ASSERT(inputs[0].nbDims == 3);
    ASSERT(inputs[1].nbDims == 3);
    // scoresSize: number of scores for one sample
    scoresSize = inputs[1].d[2]->getConstantValue();
    // boxesSize: number of box coordinates for one sample
    boxesSize = scoresSize * 4;
    // num_detections
    DimsExprs ret;
    ret.nbDims = 1;
    ret.d[0] = exprBuilder.constant(param.keepTopK);
    return ret;
}

// size_t BatchedNMSPlugin::getWorkspaceSize(int maxBatchSize) const
size_t NonMaxSuppressionPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const
{
    return detectionInferenceWorkspaceSize(param.shareLocation, /*maxBatchSize=*/1, boxesSize, scoresSize, param.numClasses,
        numPriors, param.topK, DataType::kFLOAT, DataType::kFLOAT);
}

// int BatchedNMSPlugin::enqueue(
//     int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
int NonMaxSuppressionPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, 
                const void* const* inputs, void* const* outputs, 
                void* workspace, 
                cudaStream_t stream) 
{
    const void* const locData = inputs[0];
    const void* const confData = inputs[1];

    void* nmsedIndices = outputs[0];
    pluginStatus_t status = nonMaxSuppressionInference(stream, 1, boxesSize, scoresSize, param.shareLocation,
        param.backgroundLabelId, numPriors, param.numClasses, param.topK, param.keepTopK, param.scoreThreshold,
        param.iouThreshold, DataType::kFLOAT, locData, DataType::kFLOAT, confData, nmsedIndices,
        workspace, param.isNormalized, false);
    // if (scoresSize > 1000) {
    //     std::cout << "scoresSize: " << scoresSize << ", topK: " << param.topK << ", numClasses: " << param.numClasses << ", keepTopK: " << param.keepTopK << ", scoreThreshold: " << param.scoreThreshold << ", iouThreshold: " << param.iouThreshold << std::endl;
    //     float* a = (float*)malloc(20 * 4 * sizeof(float));
    //     cudaMemcpy(a, locData, 20 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    //     for (int i = 0; i < 20; i ++) {
    //         for (int j = 0; j < 4; j ++) {
    //             std::cout << a[i * 4 + j] << ", ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    //     free(a);
    // }

    ASSERT(status == STATUS_SUCCESS);
    return 0;
}

size_t NonMaxSuppressionPlugin::getSerializationSize() const
{
    // NMSParameters, boxesSize,scoresSize,numPriors
    return sizeof(NMSParameters) + sizeof(int) * 3 + sizeof(bool);
}

void NonMaxSuppressionPlugin::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, param);
    write(d, boxesSize);
    write(d, scoresSize);
    write(d, numPriors);
    write(d, mClipBoxes);
    ASSERT(d == a + getSerializationSize());
}

// void BatchedNMSPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
//     const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
//     const bool* outputIsBroadcast, nvinfer1::PluginFormat format, int maxBatchSize)
void NonMaxSuppressionPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, 
             const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    ASSERT(nbInputs == 2);
    ASSERT(nbOutputs == 1);
    ASSERT(in[0].desc.dims.nbDims == 3);
    ASSERT(in[1].desc.dims.nbDims == 3);

    scoresSize = in[1].desc.dims.d[2];
    boxesSize = scoresSize * 4;
    // num_boxes
    numPriors = in[1].desc.dims.d[2];
    const int numLocClasses = param.shareLocation ? 1 : param.numClasses;
    // Third dimension of boxes must be either 1 or num_classes
    ASSERT(in[0].desc.dims.d[2] == 4 || in[0].desc.dims.d[2] == -1);
}


bool NonMaxSuppressionPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) 
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));
    if (pos <= 1) {
        return ((inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF)
            && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR && inOut[pos].type == inOut[0].type);
    } else {
        return inOut[pos].format == nvinfer1::PluginFormat::kLINEAR && inOut[pos].type == DataType::kINT32;
    }
}

const char* NonMaxSuppressionPlugin::getPluginType() const
{
    return NMS_PLUGIN_NAME;
}

const char* NonMaxSuppressionPlugin::getPluginVersion() const
{
    return NMS_PLUGIN_VERSION;
}

void NonMaxSuppressionPlugin::destroy()
{
    delete this;
}

IPluginV2DynamicExt* NonMaxSuppressionPlugin::clone() const
{
    auto* plugin = new NonMaxSuppressionPlugin(param);
    plugin->boxesSize = boxesSize;
    plugin->scoresSize = scoresSize;
    plugin->numPriors = numPriors;
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->setClipParam(mClipBoxes);
    return plugin;
}

void NonMaxSuppressionPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* NonMaxSuppressionPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

nvinfer1::DataType NonMaxSuppressionPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    ASSERT(index >= 0 && index < this->getNbOutputs());
    return nvinfer1::DataType::kINT32;
}

void NonMaxSuppressionPlugin::setClipParam(bool clip)
{
    mClipBoxes = clip;
}

NonMaxSuppressionPluginCreator::NonMaxSuppressionPluginCreator()
    : params{}
{
    mPluginAttributes.emplace_back(PluginField("shareLocation", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("backgroundLabelId", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("numClasses", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("topK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("keepTopK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("scoreThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("iouThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("isNormalized", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("clipBoxes", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* NonMaxSuppressionPluginCreator::getPluginName() const
{
    return NMS_PLUGIN_NAME;
}

const char* NonMaxSuppressionPluginCreator::getPluginVersion() const
{
    return NMS_PLUGIN_VERSION;
}

const PluginFieldCollection* NonMaxSuppressionPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2DynamicExt* NonMaxSuppressionPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    mClipBoxes = true;

    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "shareLocation"))
        {
            params.shareLocation = *(static_cast<const bool*>(fields[i].data));
        }
        else if (!strcmp(attrName, "backgroundLabelId"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.backgroundLabelId = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "numClasses"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.numClasses = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "topK"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.topK = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "keepTopK"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.keepTopK = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "scoreThreshold"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.scoreThreshold = *(static_cast<const float*>(fields[i].data));
        }
        else if (!strcmp(attrName, "iouThreshold"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.iouThreshold = *(static_cast<const float*>(fields[i].data));
        }
        else if (!strcmp(attrName, "isNormalized"))
        {
            params.isNormalized = *(static_cast<const bool*>(fields[i].data));
        }
        else if (!strcmp(attrName, "clipBoxes"))
        {
            mClipBoxes = *(static_cast<const bool*>(fields[i].data));
        }
    }

    NonMaxSuppressionPlugin* plugin = new NonMaxSuppressionPlugin(params);
    plugin->setClipParam(mClipBoxes);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2DynamicExt* NonMaxSuppressionPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call NMS::destroy()
    NonMaxSuppressionPlugin* plugin = new NonMaxSuppressionPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
