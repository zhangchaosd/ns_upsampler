import os

import cv2

ORI_FOLDER = "./dataset/ori_hr_videos"
HR_FOLDER = "./dataset/hr_videos"
LR_FOLDER = "./dataset/lr_videos"
NUM_FRAMES_PER_VIDEO = 100


def half_resolution(input_video):
    input_video_path = os.path.join(ORI_FOLDER, input_video)
    output_hr_video_path = os.path.join(HR_FOLDER, input_video)
    output_lr_video_path = os.path.join(LR_FOLDER, input_video)

    cap = cv2.VideoCapture(input_video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    half_frame_width = frame_width // 2
    half_frame_height = frame_height // 2
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_videos = ((frame_count - 1) // NUM_FRAMES_PER_VIDEO) + 1
    for i in range(num_videos):
        output_hr_video_path = os.path.join(
            HR_FOLDER, input_video.split(".")[0] + f"_{i}.mp4"
        )
        output_lr_video_path = os.path.join(
            LR_FOLDER, input_video.split(".")[0] + f"_{i}.mp4"
        )
        # cap.set(cv2.CAP_PROP_POS_FRAMES, i * NUM_FRAMES_PER_VIDEO)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_hr = cv2.VideoWriter(
            output_hr_video_path, fourcc, fps, (frame_width, frame_height)
        )
        out_lr = cv2.VideoWriter(
            output_lr_video_path, fourcc, fps, (half_frame_width, half_frame_height)
        )
        for _ in range(NUM_FRAMES_PER_VIDEO):
            ret, frame = cap.read()
            if ret == True:
                out_hr.write(frame)
                frame = cv2.resize(frame, (half_frame_width, half_frame_height))
                out_lr.write(frame)
            else:
                break
        out_hr.release()
        out_lr.release()
    cap.release()


def main():
    videos = [file for file in os.listdir(ORI_FOLDER) if file.endswith("mp4")]
    list(map(half_resolution, videos))


if __name__ == "__main__":
    main()
