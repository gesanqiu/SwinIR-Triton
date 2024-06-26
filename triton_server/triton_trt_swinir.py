from typing import Tuple

import tritonclient.grpc.aio as grpcclient
import numpy as np


class SwinIRTrintonClient:
    def __init__(
        self,
        triton_server_host: str,
        triton_server_port: int,
        triton_model_name: str,
        input_shape: Tuple[int, ...],
        scale: int,
        window_size: int,
    ):
        self.client = grpcclient.InferenceServerClient(f"{triton_server_host}:{triton_server_port}")
        self.model_name = triton_model_name
        self.input_shape = input_shape
        self.input_data = None
        self.inputs = []

        self.scale = scale
        self.window_size = window_size

    async def __exit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()
        if exc_type:
            print(f"Error: {exc_val}")
        return False

    async def heartbeat(self):
        input_data = np.zeros(self.input_shape, dtype=np.float32)
        self.input_data = grpcclient.InferInput("input_0", input_data.shape, "FP32")
        self.input_data.set_data_from_numpy(input_data)
        self.inputs = [self.input_data]

        try:
            response = await self.client.infer(model_name=self.model_name, inputs=self.inputs)
            # response = self.client.infer(model_name=self.model_name, inputs=self.inputs)
            output = response.as_numpy('output_0')
        except Exception as e:
            print(f"Error: {e}")
            raise e

    async def inference(self, input_data: np.ndarray):
        self.input_data = grpcclient.InferInput("input_0", input_data.shape, "FP32")
        self.input_data.set_data_from_numpy(input_data)
        self.inputs = [self.input_data]

        try:
            response = await self.client.infer(model_name=self.model_name, inputs=self.inputs)
            # response = self.client.infer(model_name=self.model_name, inputs=self.inputs)
        except Exception as e:
            print(f"Error: {e}")
            raise e

        output = response.as_numpy('output_0')

        return output
