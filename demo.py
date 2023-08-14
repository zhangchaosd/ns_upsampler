import cv2

# 使用默认摄像头（通常是0，如果你有多个摄像头，可以更改此值）
cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution: {width}x{height}")

# 设置摄像头的宽度和高度
res = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# print("set ", res)
width = cap.get(cv2.CAP_PROP_FPS)
height = cap.get(cv2.CAP_PROP_FORMAT)
print(f"Camera resolution: {width}x{height} {cap.get(cv2.CAP_PROP_FOURCC)}")
exit()

while True:
    # 从摄像头读取帧
    ret, frame = cap.read()

    # 如果帧正确读取，则显示
    if ret:
        cv2.imshow('Live Video Feed', frame)
        print(frame.shape)

        # 如果按下"q"键，退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Error reading frame")
    break

# 释放摄像头并关闭所有OpenCV窗口
cap.release()
cv2.destroyAllWindows()
