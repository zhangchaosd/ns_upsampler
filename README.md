# NS_Upsampler

This is a super resolution application for ns gaming. It's designed for enhance the resolution from 1080p to 2160p. Or enhance the quality in theory(by model training data).

<!-- ![PC](/Assets/cp0.png "PC") -->
![PC](/Assets/cp1.png "PC")
<!-- ![PC](/Assets/cp2.png "PC") -->
<!-- ![PC](/Assets/cp3.png "PC") -->

To use this, you should have a capture card hardware to project the picture to PC.

`NS -> Dock -> Capture card -> PC -> Monitor`

The first version available here: https://github.com/zhangchaosd/ns_upsampler/releases/tag/alpha

System requirments:

GPU: Intel GPU

TODO:
NVIDIA GPUs

## 1. Train a Super-Resolution model

This step is to get the `SRNet.onnx` file. You can just use the file in the repo and go to Step 2.

### 1.1 Prepareing training data

- Use capture card and Obs Studio(or other software) to record `raw` 1080p gaming videos, put these videos to `SRNet/raw_videos`.

- Run `python SRNet/prepare_data.py` to extract frames from videos.

### 1.2 Training

You can change the hyper-parameters or the model as you like. In this repo, it's a very simple small model to ensure the low infer latency.

`python SRNet/train.py`

### 1.3 Test model (optional)



## 2. Compile the NS_Upsampler application

TODO

Download OpenCV and OpenVINO sdk, put them in the folder like this:

![PC](/Assets/sdks.png "PC")

Visual Studio 2022

Open `NS_SuperResolution/NS_SuperResolution.sln` and build.

Remember copy OpenCV and OpenVINO dlls to the exe path to run.
