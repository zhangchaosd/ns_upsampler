# NS_Upsampler

This is a super resolution application for ns gaming. It's designed for enhance the resolution from 1080p to 2160p. Or enhance the quality in theory(by model training data).

To use this, you should have a capture card hardware to project the picture to PC.

`ns -> dock -> capture card -> PC -> monitor`

TODO: demo pics to add

## 1. Train a Super-Resolution model

This step is to get the `SRNet.onnx` file. You can just use the file in the repo and go to Step 2.

### 1.1 Prepareing training data

- Use capture card and Obs Studio(or other software) to record `raw` 1080p gaming videos, put these videos to `SRNet/raw_videos`.

- Run `python SRNet/prepare_data.py` to extract frames from videos.

### 1.2 Training

You can change the hyper-parameters or the model as you like. In this repo, it's a very simple small model to ensure the low infer latency.

`python SRNet/train_model.py`

### 1.3 Test model (optional)



## 2. Compile the NS_Upsampler application


Developping
