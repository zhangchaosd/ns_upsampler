import cv2
import onnx
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
# 加载模型
model = onnx.load("SRNet.onnx")

# 使用ONNX的check_model函数验证模型
onnx.checker.check_model(model)


# 加载运行时会话
sess = ort.InferenceSession('SRNet5.onnx', providers=["CUDAExecutionProvider"])

def parse_img(img_path):
    image_np = cv2.imread(img_path)  # BGR (h, w, 3)
    alpha_channel = np.full((1080, 1920, 1), 255, dtype=np.uint8)
    bgra_tensor = np.concatenate([image_np, alpha_channel], axis=-1)  # BGRA

    # 进行推断
    bgra_array = sess.run(["modelOutput"], {"modelInput":bgra_tensor})[0]
    cv2.imwrite(img_path[:-4] + "_hrr.png", bgra_array)
    print("Image saved")

# 打印输出
#parse_img("test_img.PNG")
parse_img("test1.PNG")
parse_img("test2.PNG")
parse_img("test3.PNG")
