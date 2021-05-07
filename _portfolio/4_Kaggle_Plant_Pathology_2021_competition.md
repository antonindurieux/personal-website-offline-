---
title: 'Kaggle Plant Pathology 2021 competition'
excerpt: 'Identifying the category of foliar diseases in apple trees thanks to a CNN implemented with Keras and TensorFlow, on TPU hardware.'
header:
  overlay_image: /assets/images/plant_pathology_cover.png
  show_overlay_excerpt: true
  teaser: assets/images/plant_pathology_cover.png
classes: wide
---

This article is based on the solution I submitted for the Kaggle [Plant Pathology 2021 challenge](https://www.kaggle.com/c/plant-pathology-2021-fgvc8){:target='_blank'}, which took place from March 15 2021 to May 27 2021. This competition was part of the Fine-Grained Visual Categorization [FGVC8](https://sites.google.com/view/fgvc8) workshop at the Computer Vision and Pattern Recognition Conference [CVPR 2021](http://cvpr2021.thecvf.com/){:target='_blank'}.

This competition was a good opportunity to practice and deepen my knowledge on [convolutional neural networks (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network){:target='_blank'} implementation, and to explore some technical topics such as :
- How to implement a CNN taking advantage of [TPUs](https://www.kaggle.com/docs/tpu){:target='_blank'} to speed up the computing steps ;
- How to build a Dataset from our data with TensorFlow

## 1. Introduction

### 1.1 Task
As stated on the [competition description page](https://www.kaggle.com/c/plant-pathology-2021-fgvc8/overview/description)