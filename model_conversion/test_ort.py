import argparse
import time

import numpy as np
import onnxruntime
import cv2
import torch

def get_input_shape(binding_dims):
    if len(binding_dims) == 4:
        return tuple(binding_dims[2:])
    elif len(binding_dims) == 3:
        return tuple(binding_dims[1:])
    else:
        raise ValueError('bad dims of binding %s' % (str(binding_dims)))


class Processor():
    def __init__(self, model):
        # load onnx engine
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_session = onnxruntime.InferenceSession(model, sess_options, providers=['CUDAExecutionProvider'])

        # get output name
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_names = []
        for i in range(len(self.ort_session.get_outputs())):
            output_name = self.ort_session.get_outputs()[i].name
            print("output name {}: ".format(i), output_name)
            output_shape = self.ort_session.get_outputs()[i].shape
            print("output shape {}: ".format(i), output_shape)
            self.output_names.append(output_name)

        self.input_shape = get_input_shape(self.ort_session.get_inputs()[0].shape)
        print('self.input: ', self.input_name, self.input_shape)


    def inference(self, img):
        # forward model
        res = self.ort_session.run(self.output_names, {self.input_name: img})

        # Return only the host outputs.
        return res[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Export SwinIR to ONNX weight.', add_help=False)
    parser.add_argument('--model-path', type=str, required=True, metavar="FILE", help='path to model file')
    parser.add_argument('--image-path', type=str, required=True, metavar="FILE", help='path to input image file')
    parser.add_argument('--output-path', type=str, required=True, help='path to output image file')
    parser.add_argument('--window-size', type=int, default=8, help='window size')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8')
    args = parser.parse_args()

    processor = Processor(args.model_path)
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

    output = output[..., :h_old * args.scale, :w_old * args.scale]
    output = output.squeeze().astype(np.float32).clip(0, 1)
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HWC-BGR
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    cv2.imwrite(args.output_path, output)
