# deep_blur_detection_and_classification

Tensorflow implementation of "Defocus and Motion Blur Detection with Deep Contextual Features"

For image examples:

![input2](./input/out_of_focus0607.jpg) ![output2](./output/out_of_focus0607.png)

This repository contains a test code and sythetic dataset, which consists of scenes including motion and defocus blurs together in each scene.

--------------------------

## Prerequisites (tested)
- Ubuntu 16.04
- Tensorflow 1.6.0 (<= 1.9.0)
- Tensorlayer 1.8.2
- OpenCV2

## Train Details
- We used [CUHK blur detection dataset](http://www.cse.cuhk.edu.hk/~leojia/projects/dblurdetect/dataset.html) for training our network and generating our synthetic dataset
- Train and test set lists are uploaded in 'dataset' folder

## Test Details
- download [model weights](https://drive.google.com/open?id=1gaUmaZttnXB9Ya1JmM7jOsUeUeSPIvVj) from google drive and save the model into 'model' folder.
- specify a path of input folder in 'main.py' at line #39
- run 'main.py'

```bash
python main.py
```
## Synthetic Dataset
- download [synthetic train set](https://drive.google.com/open?id=1LPaHkuQXziBWqEsM4cIwzzkcLxbupID1)(337MB) and [synthetic test set](https://drive.google.com/open?id=1wEhXlvq1wHO05HjtbDXDqnGu2q-ZFEsQ)(11.5MB) from google drive
- Note that sharp pixels, motion-blurred pixels, and defocus-blurred pixels in GT blur maps are labeled as 0, 100, and 200, respectively, in the [0,255] range.
