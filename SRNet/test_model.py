import onnx
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import numpy as np
# 加载模型
model = onnx.load("SRNet.onnx")

# 使用ONNX的check_model函数验证模型
onnx.checker.check_model(model)


# 加载运行时会话
sess = ort.InferenceSession('SRNet.onnx')

def parse_img(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transforms.functional.resize(img, (1080, 1920))
    img = transforms.functional.to_tensor(img)
    img = img.unsqueeze(0).numpy()

    # 进行推断
    img_hr = sess.run(["modelOutput"], {"modelInput":img})[0]
    print(img_hr.shape, img_hr.max(), img_hr.min())
    img_hr = img_hr.transpose(2, 3, 1, 0).squeeze()
    img_hr = (img_hr * 255).astype(np.uint8)
    img_hr = Image.fromarray(img_hr)
    img_hr.save(img_path[:-4] + "_hr1.png")
    print("Image saved")

# 打印输出
parse_img("test_img.PNG")
