import cv2
import onnxruntime as ort

model_path = 'SRNet.onnx'

sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
# sess = ort.InferenceSession('SRNet.onnx', providers=["CUDAExecutionProvider"])
print("Available Providers: ", ort.get_available_providers())

def parse_img(img_path):
    image_np = cv2.imread(img_path)  # BGR (h, w, 3)
    bgra_array = sess.run(["modelOutput"], {"modelInput":image_np})[0]
    cv2.imwrite(img_path[:-4] + "_hrr.png", bgra_array)
    print("Image saved")

def parse_img_3channel(img_path):
    image_np = cv2.imread(img_path)  # BGR (h, w, 3)
    print(image_np.shape, image_np.dtype)
    # return
    bgra_array = sess.run(["output"], {"input":image_np})[0]
    cv2.imwrite(img_path[:-4] + "_hr.png", bgra_array)
    print("Image saved")


parse_img_3channel("Assets/test1.PNG")
parse_img_3channel("Assets/test2.PNG")
parse_img_3channel("Assets/test3.PNG")
