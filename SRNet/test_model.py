import cv2
import onnxruntime as ort
import numpy as np



# 加载运行时会话
sess = ort.InferenceSession('SRNet.onnx', providers=["CPUExecutionProvider"])
# sess = ort.InferenceSession('SRNet.onnx', providers=["CUDAExecutionProvider"])

def parse_img(img_path):
    image_np = cv2.imread(img_path)  # BGR (h, w, 3)
    alpha_channel = np.full((1080, 1920, 1), 255, dtype=np.uint8)
    bgra_tensor = np.concatenate([image_np, alpha_channel], axis=-1)  # BGRA

    # 进行推断
    bgra_array = sess.run(["modelOutput"], {"modelInput":bgra_tensor})[0]
    cv2.imwrite(img_path[:-4] + "_hrr.png", bgra_array)
    print("Image saved")

def parse_img_3channel(img_path):
    image_np = cv2.imread(img_path)  # BGR (h, w, 3)
    print(image_np.shape, image_np.dtype)
    # return
    bgra_array = sess.run(["output"], {"input":image_np})[0]
    cv2.imwrite(img_path[:-4] + "_hrr.png", bgra_array)
    print("Image saved")

# 打印输出
#parse_img("test_img.PNG")
parse_img_3channel("test1.PNG")
parse_img_3channel("test2.PNG")
parse_img_3channel("test3.PNG")
