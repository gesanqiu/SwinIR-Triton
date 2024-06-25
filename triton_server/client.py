import requests
import base64

import numpy as np
import cv2

url = "http://0.0.0.0:8888/upScale"

# 图片文件路径
file_path = "256x256.bmp"
output_path = "256x256_realSR_4x_triton.jpg"


def encode_bmp_to_base64(image_path: str):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


def encode_random_image_to_base64(height: int, width: int):
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    _, encode_img = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    encoded_img_bytes = encode_img.tobytes()
    encoded_img_str = base64.b64encode(encoded_img_bytes).decode('utf-8')
    return encoded_img_str


def send_predict_request(image_path, telephoto_value):
    # Encode the image to base64
    encoded_image = encode_bmp_to_base64(image_path)
    # encoded_image = encode_random_image_to_base64(437, 550)

    # Create the request payload
    payload = {
        "bitmap": encoded_image,
        "telephoto": telephoto_value
    }

    # Send the POST request
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        data = response.json()
        encoded_img_str = data["bitmap"]
        scale = data["upScale"]

        img_bytes = base64.b64decode(encoded_img_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(output_path, img)

        print(response.elapsed.total_seconds())
        print(f"size: {img.size}, upScale: {scale}")
    else:
        print("Request failed:", response.status_code, response.text)

send_predict_request(file_path, 60)
