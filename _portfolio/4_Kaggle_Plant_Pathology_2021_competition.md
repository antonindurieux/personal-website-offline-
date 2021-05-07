---
title: 'Kaggle Plant Pathology 2021 competition'
excerpt: 'Identifying the category of foliar diseases in apple trees thanks to a CNN implemented with Keras and TensorFlow, on TPU hardware.'
header:
  overlay_image: /assets/images/plant_pathology_cover.png
  show_overlay_excerpt: true
  teaser: assets/images/plant_pathology_cover.png
classes: wide
---

## 1. Introduction

This article is based on the solution I submitted for the Kaggle [Plant Pathology 2021 challenge](https://www.kaggle.com/c/plant-pathology-2021-fgvc8){:target='_blank'}, which took place from March 15 2021 to May 27 2021. This competition was part of the Fine-Grained Visual Categorization [FGVC8](https://sites.google.com/view/fgvc8) workshop at the Computer Vision and Pattern Recognition Conference [CVPR 2021](http://cvpr2021.thecvf.com/){:target='_blank'}.

This competition was a good opportunity to practice and deepen my knowledge on [convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network){:target='_blank'} implementation, and to explore some technical topics such as :
- How to implement a CNN taking advantage of [TPUs](https://www.kaggle.com/docs/tpu){:target='_blank'} to speed up the computing steps ;
- How to build an efficient TensorFlow input pipeline with the [tf.data API](https://www.tensorflow.org/guide/data){:target='_blank'} ;
- How to take advantage of a pre-trained neural network with [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning){:target='_blank'} to achieve high performance.

### 1.1 Task

As stated on the [competition description page](https://www.kaggle.com/c/plant-pathology-2021-fgvc8/overview/description) : 

> "Apples are one of the most important temperate fruit crops in the world. Foliar (leaf) diseases pose a major threat to the overall productivity and quality of apple orchards. The current process for disease diagnosis in apple orchards is based on manual scouting by humans, which is time-consuming and expensive."

The task of this challenge was thus to develop a machine learning-based model to identify diseases on images of apple tree leaves. 

Each leaf could be healthy, or present a combination of various diseases. As each image could potentially be associated with several labels (in case of multiple diseases), this was a **multi-label classification** task.

### 1.2 Data

For the purpose of the competition, a dataset of **18632** labeled apple tree leaf images was provided. 

The test set used to evaluate the participant submissions was constituted of roughly **2700** images. 

The pictures were provided in jpeg format of relatively high resolution, lots of them being 2676 x 4000 pixels, but the resolution and aspect ratio could somewhat vary for some images.

### 1.3 Performance metric

The evaluation metric for this competition was the Mean F1-score. There are several ways to calculate the F1-score for multi-label targets (see the *average* parameter in the [Scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html){:target='_blank'} for exemple), leading to different results. It wasn't clearly specified what formula has been chosen for the competition.

