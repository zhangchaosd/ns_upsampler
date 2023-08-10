import cv2

def resize_video(input_path, output_path, target_size=(1920, 1080)):
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    
    # 获取视频的帧数
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 定义视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 调整帧的大小
        resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
        out.write(resized_frame)

    # 释放资源
    cap.release()
    out.release()

# 调用函数
input_video_path = 'v1.mp4'
output_video_path = 'v1_1080p.mp4'
resize_video(input_video_path, output_video_path)
