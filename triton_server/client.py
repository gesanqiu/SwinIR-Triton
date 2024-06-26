import requests
import base64

import numpy as np
import cv2

url = "http://0.0.0.0:8888/upScale"

# 图片文件路径
file_path = "20240626-095247.png"
output_path = "20240626-095247_realSR_4x_triton.jpg"


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


def read_and_resize_image(image_path, size=(256, 256)):
    # 1. 使用 OpenCV 读取图像
    img = cv2.imread(image_path)

    # 检查图像是否成功读取
    if img is None:
        raise ValueError("Image not found or unable to load.")

    # 2. 缩放图像到指定大小
    resized_img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

    # 3. 将图像编码为 .jpg 格式的字节数组
    success, encoded_img = cv2.imencode('.jpg', resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    if not success:
        raise ValueError("Image encoding failed.")

    encoded_img_bytes = encoded_img.tobytes()

    # 4. 将字节数组编码为 Base64 字符串
    encoded_img_str = base64.b64encode(encoded_img_bytes).decode('utf-8')

    return encoded_img_str
    

def send_predict_request(image_path, telephoto_value):
    # Encode the image to base64
    encoded_image = encode_bmp_to_base64(image_path)
    # encoded_image = encode_random_image_to_base64(437, 550)
    # encoded_image = read_and_resize_image(image_path)

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
        print(f"request_id: {data['request_id']}")
        print(f"receive time: {data['receive_time']}")
        print(f"response time: {data['response_time']}")
        t = data["response_time"] - data["receive_time"]
        print(f"response time: {t} ms")
        print(f"size: {img.size}, upScale: {scale}")
    else:
        print("Request failed:", response.status_code, response.text)

send_predict_request(file_path, 60)
