import os
import cv2


def save_video_frames_to_png(
    input_folder="SRNet/raw_videos", output_folder="SRNet/hr_frames"
):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 只处理视频文件（这里只检查了几种常见的视频扩展名，您可以根据需要添加更多）
        if filename.endswith((".mp4", ".avi", ".mkv", ".flv", ".mov")):
            print(f"Processing {filename}...")
            video_path = os.path.join(input_folder, filename)
            cap = cv2.VideoCapture(video_path)
            video_name = os.path.splitext(filename)[0]  # 去掉文件扩展名

            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                save_path = os.path.join(
                    output_folder, f"{video_name}_frame_{frame_id}.png"
                )
                cv2.imwrite(save_path, frame)
                frame_id += 1

            cap.release()

    print("Process completed!")


# 使用示例：
save_video_frames_to_png()
