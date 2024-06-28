import argparse
from contextlib import asynccontextmanager
import json
from typing import Dict, Tuple
from http import HTTPStatus
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import base64
import cv2
import cupy as cp
from cucim.skimage import exposure

from triton_trt_swinir import SwinIRTrintonClient
from protocol import UpScaleRequest, UpScaleResponse, ErrorResponse


triton_models: Dict[Tuple[int, int], SwinIRTrintonClient] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global triton_models
    try:
        model_configs = json.load(open(args.model_configs, 'r'))
        if "triton_models" not in model_configs:
            raise KeyError("The key 'triton_models' is missing from the configuration file.")

        for model_config in model_configs["triton_models"]:
            if "name" not in model_config or "input_shape" not in model_config:
                raise KeyError("Each model must have 'name' and 'input_shape' fields.")

            name = model_config["name"]
            # Input_shape is NCHW
            input_shape = model_config["input_shape"]
            scale = model_config.get("scale", 4)
            window_size = model_config.get("window_size", 8)

            model = SwinIRTrintonClient(
                triton_server_host=args.triton_server_host,
                triton_server_port=args.triton_server_port,
                triton_model_name=name,
                input_shape=tuple(input_shape),
                scale=scale,
                window_size=window_size,
            )
            await model.heartbeat()
            triton_models[(input_shape[2], input_shape[3])] = model
            print(f"Loading model: {model_config} success...")
    except KeyError as e:
        print(f"Configuration Error: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
    except Exception as e:
        print(f"Catched Error: {e}")
    yield


app = FastAPI(lifespan=lifespan)


def create_error_response(status_code: HTTPStatus,
                          message: str) -> JSONResponse:
    return JSONResponse(ErrorResponse(message=message,
                                      type="invalid_request_error").model_dump(),
                        status_code=status_code.value)


@app.post("/upScale")
async def compliance_detection(request: UpScaleRequest, raw_request: Request):
    st = time.time()
    bitmap = base64.b64decode(request.bitmap)
    np_arr = np.frombuffer(bitmap, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image if image.shape[2] == 1 else image[:, :, [2, 1, 0]], (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    try:
        client = None
        _, _, h_old, w_old = image.shape
        for shape, model in triton_models.items():
            if h_old <= shape[0] and w_old <= shape[1]:
                client = model
                break

        if client is None:
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         f"No model found for the input shape: {image.shape})")

        h_pad = client.input_shape[2] - h_old
        w_pad = client.input_shape[3] - w_old
        padded_image = np.pad(image, ((0, 0), (0, 0), (0, h_pad), (0, w_pad)), mode='constant', constant_values=0)
        print(f'[Request ID]: {request.request_id}, input shape: {image.shape}, padded shape: {padded_image.shape}')

        output = await client.inference(padded_image)
        output = output[..., :h_old * client.scale, :w_old * client.scale]
        output = np.squeeze(output).astype(np.float32)
        output = np.clip(output, 0, 1)
        if output.ndim == 3:
            output = np.transpose(output, (1, 2, 0))[:, :, [2, 1, 0]]

        # FIXME: Warmup cupy kernel
        if args.enable_clahe or request.camera_image:
            image_gpu = cp.asarray(output)
            equalized_image = exposure.equalize_adapthist(image_gpu, kernel_size=None, clip_limit=0.01, nbins=256)
            output = cp.asnumpy(equalized_image)

        output = (output * 255.0).round().astype(np.uint8)
        _, encode_img = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        encoded_img_bytes = encode_img.tobytes()
        encoded_img_str = base64.b64encode(encoded_img_bytes).decode('utf-8')
        et = time.time()

        response = UpScaleResponse(
            bitmap=encoded_img_str,
            upScale=client.scale,
            message="success",
            request_id=request.request_id,
            receive_time=int(st * 1000),
            response_time=int(et * 1000),
        )
        print(f'[Request ID]: {request.request_id}, time cost: {et - st}')
        return response
    except Exception as e:
        return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR,
                                         str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RESTful API server.")

    parser.add_argument("--host", type=str, default=None, help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--triton-server-host", type=str, default=None, help="Triton Server host name")
    parser.add_argument("--triton-server-port", type=int, default=8000, help="Triton Server port number")
    parser.add_argument("--enable-clahe", action="store_true", help="Enable Contrast Limited Adaptive Histogram Equalization (CLAHE)")
    parser.add_argument(
            '--model-configs',
            type=str,
            default=None,
            help='Configuration of the model, including model name, input shape, scale, window size, etc.',)
    args = parser.parse_args()

    try:
        uvicorn.run(app, host=args.host, port=args.port, timeout_keep_alive=5)
    except KeyboardInterrupt:
        print("Server stopped by user")