# SwinIR-Triton
This project aims to explore the deployment of SwiIR based on TensorRT and TritonServer.

## Overview
[SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/pdf/2108.10257) achieves state-of-the-art performance in:
- bicubic/lighweight/real-world image SR
- grayscale/color image denoising
- grayscale/color JPEG compression artifact reduction

You can find all the details on its official repository: [SwinIR: Image Restoration Using Swin Transformer](https://github.com/JingyunLiang/SwinIR?tab=readme-ov-file)
These repository only implement the inference acceleration based on TensorRT and inference server based on TritonServer. And here I take the realSR task as an example.

## Prerequisites
Your GPU(s) must be of Compute Capability 8.0 or higher. Amphere and later architectures are supported.And all my work is developed in docker containers provided by Nvidia NGC:
- Model Conversion: `nvcr.io/nvidia/pytorch:24.03-py3`
- TritonServer：`nvcr.io/nvidia/tritonserver:24.03-py3`

## Model Conversion
I finished all the model conversions in nvcr.io/nvidia/pytorch:24.03-py3, which have CUDA 12.4 and cuDNN 9.0.0, however, according to the requirements of ONNX Runtime, ONNX Runtime built with cuDNN 8.x are not compatible with cuDNN 9.x. So if you want to test ONNX weights with `CUDAExecutionProvider`, you can use `nvcr.io/nvidia/pytorch:23.08-py3` and install ONNX Runtime with this command.

### pth->onnx
According to my test, it will cost more than 200GB CPU/GPU RAM to convert the model that supports dynamic height and weight axes.
```shell
git clone https://github.com/gesanqiu/SwinIR-Triton.git
cd SwinIR-Triton
git submodule update --init --recursive

python model_conversion/export_onnx.py --task real_sr --scale 4 --model-path ./SwinIR/model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth --onnx-path ./SwinIR/model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.onnx --large-model

python model_conversion/test_ort.py --model-path SwinIR/model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.onnx --image-path 256x256.bmp --output-path 256x256_realSR_4x_ort.bmp
```

### onnx->plan
Here I provide two methods to convert ONNX weight to TensorRT weight. It will take about 6 hours to convert the engine that supports dynamic height and weight axes.

#### trtexec
```shell
trtexec --onnx=./SwinIR/model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.onnx --saveEngine=./SwinIR/model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.engine --verbose --minShapes=input_0:1x3x256x256 --optShapes=input_0:1x3x1024x1024 --maxShapes=input_0:1x3x1024x1024 --hardwareCompatibilityLevel=ampere+

trtexec --loadEngibe=./SwinIR/model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.engine --shapes=input_0:1x3x1024x1024 --verbose
```

#### Python API
```shell
# Since nvcr.io/nvidia/pytorch:24.03-py3 already installed TensorRT package, we only install cuda-python here
pip install cuda-python

python model_conversion/export_trt.py --onnx-path ./SwinIR/model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.onnx --save-engine-path ./SwinIR/model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.engine --min-input-shape 1x3x256x256 --opt-input-shape 1x3x1024x1024 --max-input-shape 1x3x1024x1024 --ampere-plus

trtexec --loadEngibe=./SwinIR/model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.engine --shapes=input_0:1x3x1024x1024 --verbose
```

## Benchmark(A40)
| Model Type                | Input Shape         | GPU Memory (MiB) | TensorRT Inference Latency (ms) |
|---------------------------|---------------------|------------------|---------------------------------|
| Dynamic Shape：[1, 3, -1, -1] | [1, 3, 256, 256]   | 35497            | 541                             |
| Dynamic Shape：[1, 3, -1, -1] | [1, 3, 512, 512]   | 36224            | 2092                            |
| Dynamic Shape：[1, 3, -1, -1] | [1, 3, 1024, 1024] | 36952            | 8687                            |
| Static Shape：[1, 3, 256, 256] | [1, 3, 256, 256]   | 1059             | 378                             |
| Static Shape：[1, 3, 512, 512] | [1, 3, 512, 512]   | 4133             | 1606                            |
| Static Shape：[1, 3, 1024, 1024] | [1, 3, 1024, 1024] | 18228            | 6668                            |


## TritonServer
To reduce the GPU memory overhead, which enables the deployment of more accessible GPUs, it is advisable to detach the dynamic shape model to several static models according to scenarios, such as 1x3x256x256 , 1x3x512x512, 1x3x1024x1024, etc. TritonServer will help to the instances management, we can benchmark the throughput performance and then decide the most cost-efficient deploy solution.

```shell
(triton) aiteam@aiteam:/data/ls/workSpace/SwinIR-Triton$ tree model_repository/
model_repository/
├── SwinIR_realSR_s64w8_4x_1024x1024
│   ├── 1
│   │   └── model.plan
│   └── config.pbtxt
├── SwinIR_realSR_s64w8_4x_256x256
│   ├── 1
│   │   └── model.plan
│   └── config.pbtxt
└── SwinIR_realSR_s64w8_4x_512x512
    ├── 1
    │   └── model.plan
    └── config.pbtxt

6 directories, 6 files
```
According above file structure, I converted 3 TensorRT engines with static input shape and provided a sample `config.pbtxt`, you can learn more details of TritonServer's configuration in this [page](https://github.com/triton-inference-server/server/blob/v2.33.0/docs/user_guide/model_configuration.md).
It requires 2 GPUs to start the TritonServer. You can modify the parameters of `config.pbtxt` according to your machine.

```shell
CUDA_VISIBLE_DEVICES=0,1 tritonserver --model-repository=./model_repository --metrics-port 9012 --grpc-port 9011 --http-port 9010
```

## Async Server
```shell
# Install following prerequisites in the nvcr.io/nvidia/pytorch:24.03-py3
apt update && apt-get install -y libgl1-mesa-glx
pip install packaging fastapi pydantic numpy opencv-python opencv-python-headless uvicorn tritonclient grpcio grpcio-tools gevent geventhttpclient aiohttp requests

# server
python ./triton_server/triton_server.py --host 0.0.0.0 --port 8888 --triton-server-host 0.0.0.0 --triton-server-port 9011 --model-configs ./triton_server/config.json

# client
python client.py
```

- This server manages multiple triton models by a hashmap, so it requires a config.json to init all the triton model instances in advance.
- This server accepts base64 encoded .bitmap format image and returns base64 encoded .jpg format image.
Note: Due to this [issue](https://github.com/triton-inference-server/server/issues/7343), I use gRPC TritonClient in triton_server.py.

## Further work
- Integrating SwinIR into TensorRT-LLM or vLLM should have better inference efficiency and higher throughput.

## Reference
If you find SwinIR useful or relevant to your research, please cite their paper:

```txt
@article{liang2021swinir,
  title={SwinIR: Image Restoration Using Swin Transformer},
  author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint arXiv:2108.10257},
  year={2021}
}
```

## License and Acknowledgement
This project is released under the Apache 2.0 license. The codes are based on [SwinIR](https://github.com/JingyunLiang/SwinIR) and [Swin-Transformer-TensorRT](https://github.com/maggiez0138/Swin-Transformer-TensorRT). Please also follow their licenses. Thanks for their awesome works.
