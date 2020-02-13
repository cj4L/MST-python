# Real-Time Salient Object Detection with a Minimum Spanning Tree
<p align="center">
  <img src="0027.png" width="350" title="input">
  <img src="salmap.png" width="350" alt="salmap">
</p>

This is a python re-implementation of a paper published in CVPR 2016, which can be downloaded in [here](https://ieeexplore.ieee.org/document/7780625)

Environment you need is: Python 3, cv2 and numpy. run demo.py you will get the salmap (like the right figure). It takes about 20 seconds in my personal laptop.

There is a c++ re-implementation which is much much faster than me in [here](https://github.com/lhaof/Real-Time-Salient-Object-Detection-with-a-Minimum-Spanning-Tree) . My whole algorithm is based on this and the variable name is the same as him.

Due to the subtle differences of the sort algorithm, few pixel value of my result is slightly different.

By the way, if you use the code, please cite the following reference:
```
@inproceedings{Tu2016Real,
  title={Real-Time Salient Object Detection with a Minimum Spanning Tree},
  author={Tu, Wei Chih and He, Shengfeng and Yang, Qingxiong and Chien, Shao Yi},
  booktitle={IEEE Conference on Computer Vision & Pattern Recognition},
  year={2016},
}
```
