import argparse
from typing import Tuple

from cuda import cudart
import tensorrt as trt


def export_trt_engine(
    onnx_path: str,
    save_engine: str,
    min_shape: Tuple[int],
    opt_shape: Tuple[int],
    max_shape: Tuple[int],
    ampere_plus: bool = False,
) -> None:
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(flag) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        print("Parsing ONNX file.")
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
        print('Completed parsing of ONNX file')
        
        config = builder.create_builder_config()
        if ampere_plus:
            config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS
        profile = builder.create_optimization_profile()
        for num_input in range(network.num_inputs):
            profile.set_shape(
                input=network.get_input(num_input).name,
                min=min_shape,
                opt=opt_shape,
                max=max_shape,
            )
        config.add_optimization_profile(profile)

        print("Building TensorRT engine. This may take a few minutes.")
        serialized_engine  = builder.build_serialized_network(network, config)
        print("Created engine success! ")

        with open(save_engine, 'wb') as f:
            f.write(serialized_engine)
        print("ONNX->TRT: Success...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Export ONNX weight to TensorRT engine.', add_help=False)
    parser.add_argument('--onnx-path', type=str, required=True, metavar="FILE", help='path to model file')
    parser.add_argument('--save-engine-path', type=str, required=True, default=None, help='path to output file')
    parser.add_argument('--ampere-plus', action='store_true', help='Hardware Compatibility Level')
    parser.add_argument('--min-input-shape', type=str, required=True, default=None, help='Build with dynamic shapes using a profile with the min shapes provided')
    parser.add_argument('--opt-input-shape', type=str, required=True, default=None, help='Build with dynamic shapes using a profile with the opt shapes provided')
    parser.add_argument('--max-input-shape', type=str, required=True, default=None, help='Build with dynamic shapes using a profile with the max shapes provided')
    args = parser.parse_args()

    min_shpae = args.min_input_shape.split('x')
    min_shpae = [int(dim) for dim in min_shpae]
    opt_shpae = args.opt_input_shape.split('x')
    opt_shpae = [int(dim) for dim in opt_shpae]
    max_shpae = args.max_input_shape.split('x')
    max_shpae = [int(dim) for dim in max_shpae]

    export_trt_engine(
        args.onnx_path,
        args.save_engine_path,
        min_shpae,
        opt_shpae,
        max_shpae,
        args.ampere_plus,
    )
