# Face detection & recognition
------------------------------------------------
## Table of Contents

* [Pre-requisites](#pre-requisites)
* [Quick Start](#quick-start)
* [Usage](#usage)
  * [Dataset](#dataset)
  * [Face Detection](#face-detection)
  * [Pre-trained Model](#pre-trained-model)
* [Sample Outputs](#sample-outputs)
* [References](#references)


## Pre-requisites

* argparse
    > pip install argparse
* opencv-python
    > pip install opencv-python
* opencv-contrib-python
    > pip install opencv-contrib-python
* Numpy
    > pip install numpy
* facenet-pytorch
    > pip install facenet-pytorch
* Pytorch
    > go to install pytorch(_version check!!)

## Quick Start

* Clone this repository: $ git clone https://github.com/jaehwan-AI/face_detection_recognition

* Run the demo:

>**image input**
```bash
$ python demo.py --image data/image/image.jpg
```

>**video input**
```bash
$ python demo.py --video data/video/video.mp4
```

>**webcam**
```bash
$ python demo.py --src 0
```

## Usage

### Dataset

We used Korean dataset that can't be disclosed for security reasons.

### Face Detection

We used MTCNN as a facial recognition technology to analyze emotions. MTCNN uses image pyramids by resizing images entered on different scales to recognize faces of different sizes in the images.

### Pre-trained Model

In order to inference the model, we used pre-learned weights using EfficientNet(2019).


## Sample Outputs

sample image:

<img src="sample/sample1.jpg" width="60%">

sample video:

<img src="sample/sample2.gif" width="60%">

sample webcam:


## References

1. Tim Esler's facenet-pytorch repo: [https://github.com/timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)

1. Octavio Arriaga's pre-trained model repo: [https://github.com/oarriaga/face_classification](https://github.com/oarriaga/face_classification)

1. K. Zhang, Z. Zhang, Z. Li and Y. Qiao. _Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks_, IEEE Signal Processing Letters, 2016. [PDF](https://arxiv.org/abs/1604.02878)

1. M. Tan, Quoc V. Le. _EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks_, 2019. [PDF](https://arxiv.org/abs/1610.02357)
