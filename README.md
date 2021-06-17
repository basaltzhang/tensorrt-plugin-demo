[toc]
# 背景

在实际工作当中，训练的模型到实际使用还需要有模型加速过程，比如剪枝，替换backbone，蒸馏等方法。本文主要在硬件级别对模型进行加速。

TensorRT是NVIDIA专门针对自家显卡做深度学习推理加速的框架，可为深度学习推理应用提供低延迟和高吞吐量。Pytorch是FAIR代言的训练工具，其简单易用的特点使得其成为用户数增长最快的深度学习训练框架，越来越多的学术论文放出来的源码是使用Pytorch训练。

转换模型的目的是在对应硬件上达到提速降耗效果。目前各个公司都推出了自己的框架，训练模型常用的如Facebook的Pytorch，Google的TensorFlow，Amazon的Mxnet，只有前向的如Microsoft的ONNX，NVIDIA的TensorRT，Intel的OpenVINO。各个公司的开发速度有快有慢，比如pytorch支持的功能onnx不支持，tensorrt又要与各个版本的onnx做对应开发，本文的作用就是对各个框架的协调一致，达到最终的加速目标。

转换过程也是一个学习的过程。首先需要对几个框架有所了解，如果出现问题，可以在对应框架的官方文档、github的issue区或者Google搜索。过程需要耐心和细心，涉及到python、c++、cuda、cmakefile等相关知识，通过这次转换，也可以加深自己对这几种工具的理解程度，对以后的工作也会有帮助作用。

本文使用的方法为利用ONNX作为中间框架，先将pytorch模型转为onnx模型，然后再转为tensorrt模型。勇于探索的同学可以尝试直接使用[torch2trt项目](https://github.com/NVIDIA-AI-IOT/torch2trt)。在pytorch和tensorrt的官方文档中对这两步转换的说明以及示例比较清晰，本文主要对文档外的一些特殊情况做些补充说明。

# 准备工作

先介绍一下本文使用到的软件版本，建议使用anaconda3，并升级gcc>=4.9。本文中使用torch===1.4.0，torchvision===0.5.0，cuda===10.0，gcc===5.2.0，onnxruntime。安装tensorrt，参考[文档](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)。配合CUDA==10.0版本，我使用7.0版本的tensorrt，下载后配置环境变量，并安装对应版本的tensorrt-python和pycuda。

在shell界面运行下面命令，可以查看版本号是否与上面一致，第五项输出应该为True。如果使用更高版本pytorch，需搭配的nvidia显卡驱动版本高于发布平台使用的版本。但如果是外部环境可以使用高版本pytorch以及显卡驱动，最新的pytorch中转onnx部分更新速度非常快，对转换过程更加友好。

```bash
python --version
nvcc --version
gcc --version
python -c "import torch;print(torch.__version__)"
python -c "import tensorrt as trt;print(trt.__version__)"
python -c "import torch;print(torch.cuda.is_available())"
```

# Pytorch->ONNX

主要参考[Pytorch文档](https://pytorch.org/docs/master/onnx.html)，分两种转换模式，第一种是trace-based，不支持动态网络，输入维度固定。第二种是script-based，支持动态网络，输入可变维度。

很显然第二种更加合理，但改起来相对比较复杂，这里面我们使用trace-based转换模式。另外Torchvision内部所有模型也全部支持转换到ONNX，参考文档见[这里](https://pytorch.org/docs/master/torchvision/models.html#faster-r-cnn)。

```python
import torch
import torchvision
import numpy as np

model = torchvision.models.alexnet(pretrained=True).cuda()
model.eval()
x = torch.rand(1, 3, 224, 224).to("cuda")
with torch.no_grad():
    predictions = model(x)
trace_backbone = torch.jit.trace(model, x, check_trace=False)
torch.onnx.export(trace_backbone, x, "alexnet.onnx", verbose=False, export_params=True, training=False, opset_version=10, example_outputs=predictions)

# 运行onnx的示例
import onnxruntime as ort
ort_session = ort.InferenceSession('alexnet.onnx')
onnx_outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: x.cpu().numpy().astype(np.float32)})

# 校验结果
print(predictions.cpu().numpy() - onnx_outputs)
```
结果输出示例如下，可以看到差异出现在小数点后第七位数字，转换成功。
```bash
[[[-4.7683716e-07  3.5762787e-07  3.8743019e-07  1.1920929e-07
   -9.5367432e-07 -1.0728836e-06 -1.4305115e-06  5.9604645e-08
   -7.1525574e-07  3.5762787e-07  5.9604645e-08  7.7486038e-07
    1.1920929e-07 -5.9604645e-07  4.1723251e-07 -3.5762787e-07
    0.0000000e+00 -1.7881393e-07 -1.1920929e-07  1.7881393e-07
   ...
    4.7683716e-07 -2.3841858e-07  5.6624413e-07  2.3841858e-07
   -9.5367432e-07 -7.1525574e-07  2.3841858e-07  5.9604645e-07]]]
```

# ONNX->TensorRT
参考[TensorRT文档](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_onnx_python)，以及一个[SSD例子](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#uff_ssd)，我们大概可以写个简单的ONNX转换到TensorRT并实际运行的脚本。

```python
import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnxruntime as ort

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
onnx_file_path = "alexnet.onnx"
engine_file_path = "alexnet.trt"
if os.path.exists(engine_file_path):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
else:
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                exit()
        print('Building an engine from file temp.pb; this may take a while...')
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())

inputs = []
outputs = []
bindings = []
stream = cuda.Stream()

for binding in engine:
    print(binding)
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    dtype = trt.nptype(engine.get_binding_dtype(binding))

    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)

    bindings.append(int(device_mem))

    if engine.binding_is_input(binding):
        inputs.append((host_mem, device_mem))
    else:
        outputs.append((host_mem, device_mem))

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    [cuda.memcpy_htod_async(inp[1], inp[0], stream) for inp in inputs]
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out[0], out[1],stream) for out in outputs]
    stream.synchronize()
    return [out[0] for out in outputs]

# 运行onnx的示例
x = np.random.rand(1, 3, 224, 224).astype(np.float32)
ort_session = ort.InferenceSession(onnx_file_path)
onnx_outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: x})

# 校验结果
with engine.create_execution_context() as context:
    np.copyto(inputs[0][0], x.reshape(-1))
    output = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    pred = np.array(output)
    print(onnx_outputs - pred)
```

结果也与上面类似，精确到小数点后六位左右。到这里，pytorch模型转为tensorrt模型就转换完成。保存的alexnet.trt文件可以省去再次使用时的转化步骤。可以提供给服务正常使用。

# 复杂案例
上面我们转换的是一个非常成熟的Alexnet，但有时我们需要的模型可能稍微复杂一些，有一些自定义的层。接下来我们以一个最常见的FAIR实验室的[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)中Faster RCNN模型为例，给大家简单说明一下转换过程中可能遇到的问题以及相关的解决思路。

## 编译自定义算子并运行demo

进入*maskrcnn-benchmark*目录，执行
```shell
python setup.py build develop
```
一般此处需要注意输出第一行，是否会出现找不到nvcc的异常，如果出现请把*/usr/local/cuda/bin*目录加入到路径中。执行时间大约5分钟，完毕后会在*maskrcnn_benchmark*目录出现一个*_C.{python-version}.so*的动态链接库。

新增如下代码，下载模型[地址](https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_R_50_C4_1x.pth)到*models*文件夹，代码中的图片可以自行定义，这里我用了一张ImageNet的测试图像，模型为faster rcnn，使用resnet50作为特征提取模型，使用RoiAlign做特征池化，无FPN。新建脚本*demo/demo.py*

```python
import cv2
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

if __name__ == "__main__":
    image_name = "demo/ILSVRC2012_val_00050000.JPEG"
    config_file = "configs/e2e_faster_rcnn_R_50_C4_1x.yaml"
    weight_file = "models/e2e_faster_rcnn_R_50_C4_1x.pth"

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda", "MODEL.WEIGHT", weight_file])

    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
    )
    # load image and then run prediction
    image = cv2.imread(image_name)
    height, width = image.shape[:2]
    predictions = coco_demo.run_on_opencv_image(image)
    print(predictions.shape)

    cv2.imwrite("out.jpg", predictions)
```

修改脚本中的各个文件路径，保存后执行`python demo/demo.py`。如无异常可以查看*demo.jpg*的结果。这样我们就跑通了Resnet50为基础网络的Faster Rcnn模型。

## 转换为onnx

### 准备步骤
首先需要修改一下*maskrcnn_benchmark*工程文件，把输出值变成一个tensor，这样onnx才能够正确的读取模型输出。

首先在`maskrcnn_benchmark/config/defaults.py`中加一个标志位，表示我们是在进行转换。这个文件保存了所有的默认配置，与*configs*文件夹中yaml文件共同管理代码运行时的参数。

```python
_C.MODEL.EXPORT_ON = False
```
修改`maskrcnn_benchmark/modeling/detector/generalized_rcnn.py`文件，在开头加入

```python
import torchvision
```

在`__init__`函数中加入

```python
self.cfg = cfg
self.detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
```
在`forward`函数末尾，修改返回值类型从自定义的BoxList到torch.Tensor。并补齐一下使得返回值为定长。

```python
...
        if self.cfg.MODEL.EXPORT_ON:
            boxes = torch.stack([x.bbox for x in result], 0)
            scores = torch.stack([x.get_field("scores").unsqueeze(1) for x in result], 0)
            b = torch.cat((boxes, scores), 2)
            if not torchvision._is_tracing():
                b_size = self.detections_per_img - int(b.size(1))
                fill_zeros = torch.zeros((1, b_size, 5), dtype=torch.float, device=boxes.device)
                result = torch.cat((b, fill_zeros), 1)
            else:
                return b
...
```
### 转换代码
新建转换onnx文件使用的脚本*tools/export_onnx.py*
```python
import numpy as np
import cv2
import torch
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

if __name__ == "__main__":
    config_file = "configs/e2e_faster_rcnn_R_50_C4_1x.yaml"

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda", "MODEL.WEIGHT", "models/e2e_faster_rcnn_R_50_C4_1x.pth", "MODEL.EXPORT_ON", True])

    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
    )
    # load image and then run prediction
    image = cv2.imread("demo/ILSVRC2012_val_00050000.JPEG")
    height, width = image.shape[:2]
    # predictions = coco_demo.run_on_opencv_image(image)
    coco_demo.model.eval()
    image = cv2.resize(image, (768, 768))
    image = np.stack([image] * 1, 0)
    images = torch.from_numpy(image).to(torch.float).to("cuda").permute(0, 3, 1, 2)

    with torch.no_grad():
        features = coco_demo.model(images)

    trace_backbone = torch.jit.trace(coco_demo.model, images, check_trace=False)
    torch.onnx.export(trace_backbone, images, "models/fast_rcnn.onnx", verbose=True, export_params=True, training=False, opset_version=10, example_outputs=features)
```
保存后执行命令`python tools/export_onnx.py`。

### 异常解析

执行命令后会有较长的输出，此时有两个地方需要注意，第一个是

>/export/xxx/codes/maskrcnn-benchmark/maskrcnn_benchmark/structures/bounding_box.py:21: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
  bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)

说明此处代码会使得变量变成一个常量，不会随输入发生改变。如果真的可以不发生变化的话，我们可以不用考虑。此处是输入，应该是一个变量而不是常量，修改`maskrcnn_benchmark/structures/bounding_box.py`文件，第21行修改为
```python
...
        if not isinstance(bbox, torch.Tensor):
            bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
...
```
第二个需要注意的是报错
>Traceback (most recent call last):
  File "demo/export_onnx.py", line 33, in <module>
    torch.onnx.export(trace_backbone, images, "fast_rcnn.onnx", verbose=True, export_params=True, training=False, opset_version=10, example_outputs=features)
  File "/root/xxx/env/anaconda3/lib/python3.6/site-packages/torch/onnx/\_\_init\_\_.py", line 148, in export
    strip_doc_string, dynamic_axes, keep_initializers_as_inputs)
  File "/root/xxx/env/anaconda3/lib/python3.6/site-packages/torch/onnx/utils.py", line 66, in export
    dynamic_axes=dynamic_axes, keep_initializers_as_inputs=keep_initializers_as_inputs)
  File "/root/xxx/env/anaconda3/lib/python3.6/site-packages/torch/onnx/utils.py", line 428, in \_export
    operator_export_type, strip_doc_string, val_keep_init_as_ip)
RuntimeError: ONNX export failed: Couldn't export Python operator \_ROIAlign

说明上节中编译的`_ROIAlign`算子无法从pytorch转换为onnx，需要手动添加自定义层。
### 自定义算子
根据报错内容，找到`_ROIAlign`算子所在的文件位置，`maskrcnn_benchmark/layers/roi_align.py`。可以看到`_ROIAlign`层是一个`Function`的子类，在`forward`函数里面有一个`_C.roi_align_forward`命令，就是调用了刚才编译的动态链接库中的函数。我们需要在`_ROIAlign`类中加一个函数，参考[文档](https://pytorch.org/docs/master/onnx.html#custom-operators)，让pytorch知道这个自定义的层应该怎么转为onnx。
```python
    @staticmethod
    @parse_args('v', 'v', 'is', 'i', 'f')
    def symbolic(g, input, roi, output_size, spatial_scale, sampling_ratio):
        output_size = g.op('Constant', value_t=torch.tensor([output_size], dtype=torch.int))
        spatial_scale = g.op('Constant', value_t=torch.tensor([spatial_scale], dtype=torch.float))
        sampling_ratio = g.op('Constant', value_t=torch.tensor([sampling_ratio], dtype=torch.float))
        return g.op("MaskRcnnROIAlign", input, roi, output_size, spatial_scale, sampling_ratio)

```
这里我们告诉pytorch，如果遇到这个自定义算子，前两个输入是tensor类型，后面跟着的分别是int list/int/float类型的三个变量。最终把这个方法转换为onnx中名为`MaskRcnnROIAlign`的算子。这个算子在onnx中并真实不存在，但是因为我们不需要运行onnx程序，所以可以不用在onnx中实现这个算子，只作为一个中间过渡使用。

重新运行上面的转换脚本，我们就可以成功得到*fast_rcnn.onnx*文件。此文件由于缺少算子，所以不能用onnxruntime执行。接下来我们把onnx文件转为tensorrt。

## 转换为Tensorrt
### 转换脚本

新建*tools/convert_model.py*转换脚本

```python
import os
import torch
import tensorrt as trt
import sys

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
    return val * 1 << 30

def conver_engine(onnx_file_path, engine_file_path="", max_batch_size=1):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = GiB(max_batch_size)
        builder.max_batch_size = max_batch_size
        # Parse model file
        if not os.path.exists(onnx_file_path):
            print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
            exit(0)
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print("Completed writing Engine. Well done!")

if __name__ == "__main__":
    onnx_file_path = 'models/fast_rcnn.onnx'
    engine_file_path = "models/fast_rcnn.trt"
    conver_engine(onnx_file_path, engine_file_path)
```
保存后执行`python tools/convert_model.py`。

### 异常解析之topK

由于是verbose级别的日志输出，会得到比较详细的日志。首先会出现topk异常，开启我们的debug之旅。

>ERROR: Failed to parse the ONNX file.
In node 669 (importTopK): UNSUPPORTED_NODE: Assertion failed: inputs.at(1).is_weights()

这是个问题曾经困扰了我很长时间，一度认为需要把模型拆分为多块转换，中间使用numpy连接。通过查找pytorch源码的issue区以及各个搜索引擎，我感觉修改pytorch源码以及TensorRT源码应该可以解决这个问题。
#### 修改pytorch源码
pytorch源码位置定位方法：

 1. 使用`which python`找到python所在文件夹，比如 *~/anaconda3/bin/python*。
 2. 使用`find ~/anaconda3 -name "torch"`找到torch所在文件夹 *~/anaconda3/lib/python3.6/site-packages/torch*。

进入这个文件夹，在*onnx*文件夹下保存着pytorch转换onnx调用的文件，在上节中我们使用`opset_version=10`，所以python会优先在*symbolic_opset10.py*文件中搜索转换方法，如果未搜寻到，会依次向低版本文件中搜寻。

由于10版本`topk`在tensorrt中支持较差，需使用9版本中的`topk`进行模型转换。备份后编辑*symbolic_opset10.py*文件，注释掉`topk`函数。重新转换onnx文件。

#### 修改tensorrt源码

转换后依然有问题，这时需要重新编译TensorRT。首先找到刚才安装TensorRT的文件夹，比如*/absolute_path/env/TensorRT-7.0.0.11*，

```bash
cd /absolute_path/env/TensorRT-7.0.0.11
export TRT_RELEASE=`pwd`

# 返回到env目录
cd ..

git clone -b master https://github.com/nvidia/TensorRT TensorRT
cd TensorRT
git submodule update --init --recursive
export TRT_SOURCE=`pwd`

# 重新编译
cd $TRT_SOURCE
mkdir -p build && cd build
export CXX=/usr/local/bin/g++
cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_OUT_DIR=`pwd` -DCUDA_VERISON=10.0 -DCUDNN_VERSION=7.5 -DPROTOBUF_VERSION=3.8.0 -DBUILD_PARSERS=ON -DBUILD_PLUGINS=ON -DBUILD_SAMPLES=OFF
make -j$(nproc)
```

编译建议使用c++4.x版本，所以会在cmake上一步指定使用系统g++。在执行cmake命令时，酌情根据自己机器环境修改参数。

这里有个小tips是make会下载protobuf源码并编译，比较耗时。可以提前下载并跳过校验步骤。

1. 可以提前下载[protobuf-cpp-3.8.0.tar.gz](https://github.com/protocolbuffers/protobuf/releases/download/v3.8.0/protobuf-cpp-3.8.0.tar.gz)到*/absolute_path/env*目录，
2. `ln -sf /absolute_path/env/protobuf-cpp-3.8.0.tar.gz $TRT_SOURCE/build/third_party.protobuf/src`,
3. 编辑*$TRT_SOURCE/build/third_party.protobuf/src/third_party.protobuf-stamp/download-third_party.protobuf.cmake*文件，在82行加入`return()`。

然后将编译出来的动态库替换原tensorrt的库文件。

```bash
cd $TRT_RELEASE/lib
ln -sf $TRT_SOURCE/build/libnvinfer_plugin.so.7.0.0.1 libnvinfer_plugin.so.7.0.0
ln -sf $TRT_SOURCE/build/libnvonnxparser.so.7.0.0 libnvonnxparser.so.7.0.0
```

到这里我们就可以做到在python调用tensorrt库时，使用修改后的tensorrt源码。

接下来是修改*$TRT_SOURCE/parsers/onnx/builtin_op_importers.cpp*文件，每一项`DEFINE_BUILTIN_OP_IMPORTER`函数是解析onnx文件算子到tensorrt的映射。搜索topk，修改这个函数里面的

```c++
if (ctx->getOpsetVersion() >= 10)
```

为

```c++
if (ctx->getOpsetVersion() > 10)
```

目的是让tensorrt在转换过程中即使发现onnx文件是10版本，也用9版本的方法来解析文件的topk函数，然后到*$TRT_SOURCE/build*文件夹执行`make -j2`后即可。

#### 重新转换

重新转换onnx和tensorrt，还会报topk异常，需要修改maskrcnn-benchmark中*configs/e2e_faster_rcnn_R_50_C4_1x.yaml*文件，将`PRE_NMS_TOP_N_TEST`的值从6000改为1000后解决。

### 异常解析之NonZero

重新编译后新的报错为

> ERROR: Failed to parse the ONNX file.
> In node 669 (parseGraph): UNSUPPORTED_NODE: No importer registered for op: NonZero

说明在onnx模型文件中存在NonZero算子。此算子的主要功能是提取标量中非零值的索引，它的返回值的长度是可变的。此算子与tensorrt这种预分配固定空间的框架不符，所以在tensorrt中没有相应的转换。我们要找到*maskrcnn_benchmark*项目中哪里用到了NonZero算子，并想办法在onnx模型文件中去掉这个算子。

重新转换onnx，在日志中从前向后搜索NonZero，第一处出现的位置日志为

>   %947 : Tensor = onnx::NonZero(%946), scope: \_\_module.rpn/\_\_module.rpn.box_selector_test
>   %948 : Tensor = onnx::Transpose\[perm=[1, 0]](%947), scope: \_\_module.rpn/\_\_module.rpn.box_selector_test
>   %949 : Tensor = onnx::Squeeze\[axes=[1]](%948), scope: \_\_module.rpn/\_\_module.rpn.box_selector_test
>   %950 : Long(1) = onnx::Cast\[to=7](%949), scope: \_\_module.rpn/\_\_module.rpn.box_selector_test # /maskrcnn-benchmark/maskrcnn_benchmark/modeling/rpn/inference.py:98:0

表明是在*maskrcnn_benchmark/modeling/rpn/inference.py*文件中第98行使用的，找到这一行发现是个`torch.arange`操作。解决办法修改*symbolic_opset10.py*文件，加入arange算子

```python
def arange(g, *args):
    from torch.onnx.symbolic_opset11 import arange as arange11
    return arange11(g, *args)
```

另外此函数有个`remove_small_boxes`操作，这里会去除过小的框。在转换时可以直接注释掉，或者

```python
...
            if not torchvision._is_tracing():
                boxlist = remove_small_boxes(boxlist, self.min_size)
...
```

第三处可以看到是在*maskrcnn_benchmark/modeling/roi_heads/box_head/inference.py*文件中，filter_results函数过滤阈值低于`self.score_thresh`然后进行nms操作。由于nms算子在tensorrt中也不存在，则此处可以在后面与nms自定义层一同处理。

### 自定义算子NMS

重新执行`python tools/export_onnx.py && python tools/convert_model.py`后，出现报错如下

> [TensorRT] VERBOSE: /TensorRT/parsers/onnx/ModelImporter.cpp:129:  [Slice] inputs: [986 -> (1)], [990 -> (1)], [991 -> (1)], [992 -> (1)],
> ERROR: Failed to parse the ONNX file.
> In node 715 (importSlice): UNSUPPORTED_NODE: Assertion failed: axes.valuesKnown()

分析日志找到是在*maskrcnn_benchmark/modeling/rpn/inference.py*文件调用`boxlist_nms`函数时出现的异常。`boxlist_nms`函数会返回一个不定长的nms后的结果，而此nms算子在tensorrt中不存在。我们需要在tensorrt源码中加入这个nms自定义层。

#### 加入自定义torch层

类似前面转换onnx时，我们对自定义`_ROIAlign`层加入虚拟onnx算子，这里我们也建立一个自定义层，并加入虚拟onnx算子。由于不进行真实前向和反向操作，这两处都可以模拟输出值，只关注转换需要用到的`symbolic`函数即可。代码如下

```python
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.onnx.symbolic_opset9 import unsqueeze
from torch.onnx.symbolic_helper import parse_args

class NonMaxSuppression(Function):
    @staticmethod
    @parse_args('v', 'v', 'f', 'f', 'i')
    def symbolic(g, boxes, scores, iouThreshold, scoreThreshold=0.0, keepTopK=-1):
        boxes = unsqueeze(g, boxes, 0)
        scores = unsqueeze(g, unsqueeze(g, scores, 0), 0)
        if keepTopK == -1:
            keepTopK = boxes.size(0)
        iouThreshold = g.op('Constant', value_t=torch.tensor([iouThreshold], dtype=torch.float))
        scoreThreshold = g.op('Constant', value_t=torch.tensor([scoreThreshold], dtype=torch.float))
        keepTopK = g.op('Constant', value_t=torch.tensor([keepTopK], dtype=torch.int))
        return g.op("NonMaxSuppression", boxes, scores, iouThreshold, scoreThreshold, keepTopK)

    @staticmethod
    def forward(g, boxes, scores, iouThreshold, scoreThreshold=0.0, keepTopK=-1):
        if keepTopK == -1:
            keepTopK = boxes.size(0)
        return torch.ones(keepTopK, device=boxes.device, dtype=torch.long)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        pass
```

保存到*maskrcnn_benchmark/layers/symbolic.py*文件，然后在*maskrcnn_benchmark/layers/\_\_init\_\_.py*文件加入

```python
from .symbolic import NonMaxSuppression
```

在`__all__`变量中加入`"NonMaxSuppression"`。由于nms返回值不定长，这里面我返回了一个定长数组，用`keepTopK`指定。如果nms后的长度小于这个值，返回索引用-1来补齐，使用索引取值时就会取到概率值最低的那个框。

#### 修改torch调用

在*maskrcnn_benchmark/modeling/rpn/inference.py*文件中加入对自定义层`NonMaxSuppression`的引用：

```python
from maskrcnn_benchmark.layers import NonMaxSuppression
```

在`forward_for_single_feature_map`函数内，修改for循环内代码

```python
...
                if torchvision._is_tracing():
                    keep = NonMaxSuppression.apply(boxlist.bbox, boxlist.get_field("objectness"), self.nms_thresh, 0, self.post_nms_top_n)
                    boxlist = boxlist[keep]
                else:
                    boxlist = remove_small_boxes(boxlist, self.min_size)
                    boxlist = boxlist_nms(
                        boxlist,
                        self.nms_thresh,
                        max_proposals=self.post_nms_top_n,
                        score_field="objectness",
                    )
                result.append(boxlist)
...
```

在*maskrcnn_benchmark/modeling/roi_heads/box_head/inference.py*文件中加入对自定义层`NonMaxSuppression`的引用：

```python
import torchvision
from maskrcnn_benchmark.layers import NonMaxSuppression
```

在`filter_results`函数`result = []`后加入如下代码，由于nms使用-1补齐，这里加入了topk避免输出个数过多。

```python
...
        if torchvision._is_tracing():
            scores = torch.split(scores, 1, 1)
            for j in range(1, num_classes):
                boxes_j = boxes[:, j * 4 : (j + 1) * 4]
                scores_j = scores[j].flatten()
                idx = NonMaxSuppression.apply(boxes_j, scores_j, self.nms, self.score_thresh, self.detections_per_img)
                boxlist_for_class = BoxList(boxes_j[idx, :], boxlist.size, mode="xyxy")
                boxlist_for_class.add_field("scores", scores_j[idx])
                result.append(boxlist_for_class)
            result = cat_boxlist(result)
            objectness = result.get_field("scores")
            post_nms_top_n = min(self.detections_per_img, len(objectness))
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            return result[inds_sorted]
...
```

#### 修改TensorRT源码

重新转换onnx后，再转TensorRT时会报找不到刚才自定义的NonMaxSuppression算子

> ERROR: Failed to parse the ONNX file.
> In node 710 (parseGraph): UNSUPPORTED_NODE: No importer registered for op: NonMaxSuppression

此时需要进入`$TRT_SOURCE`文件夹，加入NonMaxSuppression自定义算子的真实实现。参考[项目](https://github.com/TrojanXu/onnxparser-trt-plugin-sample)

第一步，修改*parsers/onnx/builtin_op_importers.cpp*文件，加入转换函数，使得onnx的`NonMaxSuppression`算子能够和tensorrt的自定义算子找到对应关系。

```c++
DEFINE_BUILTIN_OP_IMPORTER(NonMaxSuppression)
{
    std::vector<nvinfer1::ITensor*> tensors;
    tensors.push_back(&convertToTensor(inputs.at(0), ctx));
    tensors.push_back(&convertToTensor(inputs.at(1), ctx));
    // input[0].shape = [num_boxes, 4]
    // input[1].shape = [num_boxes]

    LOG_VERBOSE("call nms plugin: ");
    const std::string pluginName = "NonMaxSuppression_TRT";
    const std::string pluginVersion = "1";

    std::vector<nvinfer1::PluginField> f;
    bool shareLocation = true;
    int backgroundLabelId = -1;
    int numClasses = 1;
    int topK = tensors[1]->getDimensions().d[2];
    float iouThreshold = static_cast<float*>(inputs.at(2).weights().values)[0];
    float scoreThreshold = (node.input().size() > 3) ? static_cast<float*>(inputs.at(3).weights().values)[0] : 0.;
    int keepTopK = (node.input().size() > 4) ? static_cast<int*>(inputs.at(4).weights().values)[0] : tensors[1]->getDimensions().d[2];
    std::cout << "iouThreshold: " << iouThreshold << ", scoreThreshold: " << scoreThreshold << ", keepTopK: " << keepTopK <<std::endl;
    bool isNormalized = false;
    f.emplace_back("shareLocation", &shareLocation, nvinfer1::PluginFieldType::kUNKNOWN, 1);
    f.emplace_back("backgroundLabelId", &backgroundLabelId, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("numClasses", &numClasses, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("keepTopK", &keepTopK, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("topK", &topK, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("iouThreshold", &iouThreshold, nvinfer1::PluginFieldType::kFLOAT32, 1);
    f.emplace_back("scoreThreshold", &scoreThreshold, nvinfer1::PluginFieldType::kFLOAT32, 1);
    f.emplace_back("isNormalized", &isNormalized, nvinfer1::PluginFieldType::kUNKNOWN, 1);

    // Create plugin from registry
    const auto mPluginRegistry = getPluginRegistry();
    const auto pluginCreator
        = mPluginRegistry->getPluginCreator(pluginName.c_str(), pluginVersion.c_str());

    ASSERT(pluginCreator != nullptr, ErrorCode::kINVALID_VALUE);

    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = f.size();
    fc.fields = f.data();

    auto plugin = pluginCreator->createPlugin(node.name().c_str(), &fc);

    ASSERT(plugin != nullptr && "NonMaxSuppression plugin was not found in the plugin registry!",
        ErrorCode::kUNSUPPORTED_NODE);

    auto layer = ctx->network()->addPluginV2(&tensors[0], int(tensors.size()), *plugin);
    nvinfer1::ITensor* indices = layer->getOutput(0);

    RETURN_FIRST_OUTPUT(layer);
}
```

注意此处是建立网络时运行的代码，实际做infer的时候不会使用此处代码，所以我们不能获取到那些**标量**的值，但能够获取**常量**的值。代码的一些简单说明：

- 首行的`NonMaxSuppression`对应onnx算子名称，
- 函数先把前两个输入，整合成为标量数组，注意此处的输入仅仅包含变量的属性信息，获取不到权重。
- 定义了tensorrt中对应算子的名称，`NonMaxSuppression_TRT`，此处后面用到。
- 把其他需要的参数传入到网络，因为前面转onnx时，这些参数都是constant格式，所以此处可以取到他们的值。
- 后面是获取自定义算子以及传参到网络和获取网络返回结果。

第二步，加入自定义的非极大值抑制层。在*plugin*文件夹我们看到已经存在了一个*batchedNMSPlugin*文件夹，因为输出与我们定义的后端-1补齐的索引不同，我们不能直接使用，但绝大部分可以复用。拷贝*batchedNMSPlugin*文件夹到新的*nonMaxSuppressionPlugin*文件夹，我们再做一些修改。

编辑*plugin/CMakeLists.txt*文件，在`PLUGIN_LISTS`加入我们新建的文件夹名称，nonMaxSuppressionPlugin。

参考官方文档见[这里](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#add_custom_layer)，新增的代码见这里。主要修改点如下：

- 遵循文档内容，使用`IPluginV2DynamicExt`代替了原来使用的`IPluginV2Ext`父类，修改各个成员函数返回值，如输入输出标量的个数、维度以及类型。
- 返回值改为索引值，长度为固定的`keepTopK`个，不够长时使用-1补齐

到*build*文件夹执行`make -j$(nproc)`后生成动态链接库。没有报错后在进入*maskrcnn-benchmark*项目，重新转换tensorrt，此时会发现找不到非极大值抑制的异常消失，取而代之的是找不到`RoiAlign`自定义层。

### 自定义算子RoiAlign（略）

此部分报错和上节相同，

> ERROR: Failed to parse the ONNX file.
> In node 718 (parseGraph): UNSUPPORTED_NODE: No importer registered for op: MaskRcnnROIAlign

此处也需要在tensorrt中加入自定义层和转换关系，方法逻辑与上节大致相同，不同的是此自定义层需要自行编写cuda核函数，进行并行加速。核函数可以在*maskrcnn-benchmark*项目中找到，稍加修改即可。各位可以尝试自行实现，也可以直接使用工程内我实现的方法。

同样在*build*目录`make -j$(nproc)`后可以生成包含RoiAlign算子的动态链接库。在*maskrcnn-benchmark*项目中执行`python tools/convert_model.py`，此时如无报警，需要花费约10分钟至半小时转换，生成最终的tensorrt框架下的模型文件。

## 运行tensorrt模型

### 运行脚本

这里主要考虑保证tensorrt模型与pytorch模型的输入一致，即图像的预处理问题。在maskrcnn-benchmark项目中，图像的预处理为以下步骤：

1. 使用opencv读取图像，此时维度是HWC，bgr模式，像素值[0, 255]
2. 转为pil格式图像，
3. resize
4. 转为tensor，并且像素值[0.0, 1.0]
5. 像素值缩放到[0, 255]
6. bgr通道减去均值

不同项目中预处理不一定相同，但一定要保证模型的输入是一致。所以我们有如下测试脚本。

```python
import os
import torch
import tensorrt as trt
from PIL import Image
import numpy as np
import common
from tools.convert_model import conver_engine
import time
import cv2
import glob

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
if __name__ == "__main__":
    onnx_file_path = 'models/fast_rcnn.onnx'
    engine_file_path = "models/fast_rcnn.trt"
    threshold = 0.5
    image_name = "demo/ILSVRC2012_val_00050000.JPEG"
    if not os.path.exists(engine_file_path):
        print("no engine file")
        # conver_engine(onnx_file_path, engine_file_path)
    print(f"Reading engine from file {engine_file_path}")
    preprocess_time = 0
    process_time = 0
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        with runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            image = cv2.imread(image_name)
            a = time.time()
            image_height, image_width = image.shape[:2]
            # image = cv2.resize(image, (768, 768)).transpose((2, 0, 1))
            image = np.array(cv2.resize(image, (768, 768)), dtype=np.float)
            image -= np.array([102.9801, 115.9465, 122.7717])
            image = np.transpose(image, (2, 0, 1)).ravel()
            # image_batch = np.stack([image], 0).ravel()
            np.copyto(inputs[0].host, image)
            preprocess_time += time.time() - a
            a = time.time()
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            process_time += time.time() - a
            x = trt_outputs[0].reshape((100, 5))
            # imshow
            image = cv2.imread(image_name)
            indices = x[:, -1] > threshold
            polygons = x[indices, :-1]
            scores = x[indices, -1]
            polygons[:, ::2] *= 1. * image.shape[1] / 768
            polygons[:, 1::2] *= 1. * image.shape[0] / 768

            for polygon, score in zip(polygons, scores):
                print(polygon, score)
                cv2.rectangle(image, (int(polygon[0]), int(polygon[1])), (int(polygon[2]), int(polygon[3])), color=(0, 255, 0), thickness=2)
                cv2.putText(image, str("%.3f" % score), (int(polygon[0]), int(polygon[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, False)
            cv2.imwrite("tensorrt_demo.jpg", image)

            print("preprocess time: ", preprocess_time, ", inference time: ", process_time)
```

保存为*demo/tensorrt_demo.py*，执行`python demo/tensorrt_demo.py`即可运行脚本。

### 异常解析

但此时我们得到的结果是空。首先怀疑是自定义层出现的问题，第一个使用到的自定义层是nms，我们可以在tensorrt中打印一下该层的输入。编辑*plugin/nonMaxSuppressionPlugin/nonMaxSuppressionPlugin.cpp*文件，可以在`enqueue`函数中加入测试代码。

```c++
        float* a = (float*)malloc(20 * 4 * sizeof(float));
        cudaMemcpy(a, locData, 20 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 20; i ++) {
            for (int j = 0; j < 4; j ++) {
                std::cout << a[i * 4 + j] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        free(a);
```

看到输入的bbox坐标都为0，一般这时会在maskrcnn-benchmark项目中加入断点，提前返回中间结果来排查问题。这里我们定位到问题是*maskrcnn_benchmark/modeling/box_coder.py*文件85行以后并没有赋值成功。

```python
...
        # pred_boxes = torch.zeros_like(rel_codes)
        # # x1
        # pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # # y1
        # pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        # pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        # # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        # pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

        const_0_5 = torch.tensor(0.5, dtype=pred_ctr_x.dtype)
        pred_boxes1 = pred_ctr_x - const_0_5 * pred_w
        pred_boxes2 = pred_ctr_y - const_0_5 * pred_h
        pred_boxes3 = pred_ctr_x + const_0_5 * pred_w
        pred_boxes4 = pred_ctr_y + const_0_5 * pred_h
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
...
```

修改后pytorch->onnx成功，onnx->tensorrt时会出现arange参数是浮点数的异常，修改*maskrcnn_benchmark/modeling/rpn/anchor_generator.py*文件`grid_anchors`函数中，`torch.arange`的参数为`dtype=torch.int64`。

重新转换后，nms的输入值非零，但仍与pytorch的不同。*maskrcnn_benchmark/modeling/rpn/inference.py*文件中`boxlist.clip_to_image`函数未起作用导致。于是此for循环需要进行如下改写。

```python
...
        for proposal, score, im_shape in zip(proposals, objectness, image_shapes):
            if torchvision._is_tracing():
                proposal = torch.stack((
                    proposal[:, 0].clamp(min=0, max=im_shape[0] - 1),
                    proposal[:, 1].clamp(min=0, max=im_shape[1] - 1),
                    proposal[:, 2].clamp(min=0, max=im_shape[0] - 1),
                    proposal[:, 3].clamp(min=0, max=im_shape[1] - 1),
                ), axis=1)
                boxlist = BoxList(proposal, im_shape, mode="xyxy")
                boxlist.add_field("objectness", score)
                keep = NonMaxSuppression.apply(boxlist.bbox, boxlist.get_field("objectness"), self.nms_thresh, 0, self.post_nms_top_n)
                boxlist = boxlist[keep]
            else:
                boxlist = BoxList(proposal, im_shape, mode="xyxy")
                boxlist.add_field("objectness", score)
                boxlist = boxlist.clip_to_image(remove_empty=False)
                boxlist = remove_small_boxes(boxlist, self.min_size)
                boxlist = boxlist_nms(
                    boxlist,
                    self.nms_thresh,
                    max_proposals=self.post_nms_top_n,
                    score_field="objectness",
                )
            result.append(boxlist)
...
```

至此结果与pytorch一致。

细心的读者可能会发现tensorrt的计算时间会比pytorch时间更长，原因是在box header计算时，pytorch提取box个数是nms后的个数，一般是几十个框。而在tensorrt中由于nms的补齐，是1000个框。修改*configs/e2e_faster_rcnn_R_50_C4_1x.yaml*文件中`POST_NMS_TOP_N_TEST`的值为100后，重新执行`python tools/export_onnx.py && python tools/convert_model.py && python demo/tensorrt_demo.py`，这样就会得到速度比pytorch更加快速的结果了。至此转换成功。


# 总结

- 转换过程中遇到参数问题、或者接口使用的问题推荐搜索官方文档。
- 专用软件的安装可以参考文档或者github页面的说明。
- 转onnx和tensorrt过程中的异常报错，可以试着在github对应的issue区搜索，别人大概率会遇到过类似的情况，会有对应的解决办法。
- 一些编译错误、语法问题或者常用软件的安装可以使用搜索引擎比如谷歌百度。

本文也是只针对pytorch->onnx->tensorrt这一种流程做了简单介绍，其他方法也需要继续尝试，比如TensorRT官方出了一个pytorch->tensorrt的版本，也欢迎各位同学勇于尝试新的项目，这样也能加快版本的迭代和技术的进步。 
