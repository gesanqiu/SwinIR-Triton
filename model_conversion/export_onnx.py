import os
import requests
import argparse
import sys

import torch

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_dir)
    swinir_path = os.path.join(current_dir, '../SwinIR')
    print(swinir_path)
    sys.path.append(swinir_path)
    from main_test_swinir import define_model
except Exception as e:
    print('Error: %s' % e)
    print('Please make sure you have downloaded the SwinIR repository.')
    sys.exit(1)


def export_onnx(model, output):
    try:
        dummy_input = torch.randn(1, 3, 512, 512, device='cpu')
        input_names = ["input_0"]
        output_names = ["output_0"]

        dynamic_axes = {'input_0': {2: "height", 3: "width"}, 'output_0': {2: "height", 3: "width"}}
        torch.onnx.export(model, dummy_input, output, verbose=False, opset_version=13,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=dynamic_axes,
                          do_constant_folding=True,
                        )
        print('ONNX export success, saved as %s' % output)
    except Exception as e:
        print('ONNX export failure: %s' % e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Export SwinIR to ONNX weight.', add_help=False)
    parser.add_argument('--model-path', type=str, required=True, metavar="FILE", help='path to model file')
    parser.add_argument('--onnx-path', type=str, required=True, default=None, help='path to output file')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--task', type=str, default='color_dn', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car, color_jpeg_car')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training-patch-size', type=int, default=128, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--large-model', action='store_true', help='use large model, only provided for real image sr')
    args = parser.parse_args()
    
    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(args.model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {args.model_path}')
        open(args.model_path, 'wb').write(r.content)

    model = define_model(args)

    export_onnx(model, args.onnx_path)
