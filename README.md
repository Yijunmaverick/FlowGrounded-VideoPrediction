# FlowGrounded-VideoPrediction
Torch implementation of our ECCV18 [paper](https://arxiv.org/pdf/1807.09755.pdf) on video prediction.

<p>
    <img src='examples/walk0.png' width=100 />
    <img src='examples/walk_pred16.gif' width=100 />
    <img src='examples/flag0.png' width=100 />
    <img src='examples/flag_pred16.gif' width=100 />
    <img src='examples/cloud0.png' width=100 />
    <img src='examples/cloud_pred16.gif' width=100 />
</p>

In each panel from left to right: the starting frame and predicted sequence (next 16 frames).

## Getting started

- Linux
- NVIDIA GPU
- Torch 
- [SPyNet](https://github.com/anuragranj/end2end-spynet)
- [ffmpeg](https://www.ffmpeg.org/)

```
git clone https://github.com/Yijunmaverick/FlowGrounded-VideoPrediction
cd FlowGrounded-VideoPrediction
```

## Preparation

- Data

  - Put the video data (e.g., `.mp4` or `.avi`) in a folder and put it under `./datasets/DTexture/raw/`.
  - Run the following command to convert videos to frames and generate the metadata for training. The testing data are prepared in the same way. Make sure that the meta data for both training and testing are ready before experiments.

```
cd datasets/
sh data_process.sh
cd ..
```

- SPyNet

  - We use the flows estimated by the great work [SPyNet]() as the ground truth for training. Make sure that the SPyNet [code](https://github.com/anuragranj/end2end-spynet) is complied successfully and works well.

- Models

  - Run the following command to download the pretrained VGG (for perceptual loss) and our model learned on the `WavingFlag` data for testing.

```
sh download_models.sh
```

## Training

  - Train the 3DcVAE model for flow prediction:

```
th train_3DcVAE.lua --dataRoot datasets/DTexture
```
  - Train the flow2rgb model for frame generation:

```
th train_flow2rgb.lua --dataRoot datasets/DTexture
```

## Testing

  - Test two steps (prediction + generation) together:

```
th test.lua --dataRoot datasets/DTexture
```
  - With `ffmpeg` installed, run the following command to convert frames to a gif or video:

```
python gif.py
```


## Acknowledgement

- Codes are heavily borrowed from [DrNet](https://github.com/edenton/drnet).
