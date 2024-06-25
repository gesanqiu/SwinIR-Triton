import time
import argparse

import common

import cv2
import torch
import numpy as np
from cuda import cudart
import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class Processor():
    def __init__(self, engine_path, window_size=8, scale=4):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context
        self.window_size = window_size
        self.scale = scale

    def inference(self, img):
        b, c, h_new, w_new = img.shape
        # FIXME: This is a hack for dynamic shape inputs/outpus
        self.context.set_input_shape("input_0", (b, c, h_new, w_new))
        trt_inputs = []
        trt_outputs = []
        trt_allocations = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = self.engine.get_tensor_dtype(name)   # float32
            shape = self.engine.get_tensor_shape(name)   # (1, 3, -1, -1)
            # FIXME: This is a hack for dynamic shape inputs/outpus
            if is_input:
                shape = [b, c, h_new, w_new]
                batch_size = shape[0]
            else:
                shape = [b, c, h_new*self.scale, w_new*self.scale]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
                'size': size
            }
            print(binding)
            trt_allocations.append(allocation)
            if is_input:
                trt_inputs.append(binding)
            else:
                trt_outputs.append(binding)
        assert batch_size > 0
        assert len(trt_inputs) > 0
        assert len(trt_outputs) > 0
        assert len(trt_allocations) > 0

        img = np.ascontiguousarray(img)
        common.memcpy_host_to_device(trt_inputs[0]['allocation'], img)

        self.context.execute_v2(trt_allocations)

        trt_output = np.zeros((1, 3, h_new*self.scale, w_new*self.scale), trt_outputs[0]['dtype'])
        common.memcpy_device_to_host(trt_output, trt_outputs[0]['allocation'])

        for binding in trt_allocations:
            common.cuda_call(cudart.cudaFree(binding))
        trt_inputs.clear()
        trt_outputs.clear()
        trt_allocations.clear()

        return trt_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Export SwinIR to ONNX weight.', add_help=False)
    parser.add_argument('--model-path', type=str, required=True, metavar="FILE", help='path to model file')
    parser.add_argument('--image-path', type=str, required=True, metavar="FILE", help='path to input image file')
    parser.add_argument('--output-path', type=str, required=True, help='path to output image file')
    parser.add_argument('--window-size', type=int, default=8, help='window size')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8')
    args = parser.parse_args()

    processor = Processor(args.model_path, args.window_size, args.scale)
    image = cv2.imread(args.image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    image = np.transpose(image if image.shape[2] == 1 else image[:, :, [2, 1, 0]], (2, 0, 1)) # HWC-BGR to CHW-RGB
    image = np.expand_dims(image, axis=0)

    b, c, h_old, w_old = image.shape
    h_pad = (h_old + args.window_size - 1) // args.window_size * args.window_size - h_old
    w_pad = (w_old + args.window_size - 1) // args.window_size * args.window_size - w_old
    print(h_old, w_old, h_pad, w_pad)

    # Pad the image
    padded_image = np.pad(image, ((0, 0), (0, 0), (0, h_pad), (0, w_pad)), mode='constant', constant_values=0)

    st = time.time()
    output = processor.inference(padded_image)
    torch.cuda.synchronize()
    print('Time: ', time.time() - st)

    output = output[..., :h_old*args.scale, :w_old*args.scale]
    print('output: ', output.shape)
    output = output.squeeze().astype(np.float32).clip(0, 1)
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HWC-BGR
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    cv2.imwrite(args.output_path, output)
