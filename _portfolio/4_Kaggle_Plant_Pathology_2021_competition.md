---
title: 'Kaggle Plant Pathology 2021 competition'
excerpt: 'Identifying the category of foliar diseases in apple trees thanks to a CNN implemented with Keras and TensorFlow, on TPU hardware.'
header:
  overlay_image: /assets/images/plant_pathology_cover.png
  show_overlay_excerpt: true
  teaser: assets/images/plant_pathology_cover.png
classes: wide
---

# Kaggle Plant Pathology 2021 competition

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

<p float="center">
  <center>
  <img src="/assets/images/plant_pathology_exemple1.jpg" width="200" />
  <img src="/assets/images/plant_pathology_exemple2.jpg" width="200" /> 
  <img src="/assets/images/plant_pathology_exemple3.jpg" width="200" />
  <img src="/assets/images/plant_pathology_exemple4.jpg" width="200" />
  <br>
  <em>Examples of images from the dataset</em></center>
</p>

The labels were provided in a separate csv file.

### 1.3 Performance metric

The evaluation metric for this competition was the Mean F1-score. There are several ways to calculate the F1-score for multi-label targets (see the *average* parameter in the [Scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html){:target='_blank'} for exemple), leading to different results. It wasn't clearly specified what formula has been chosen for the competition.

## 2. Imports and configuration

```python
# Imports
import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from kaggle_datasets import KaggleDatasets

sns.set()
```

### 2.1 TPU configuration

[TPUs](https://en.wikipedia.org/wiki/Tensor_Processing_Unit){:target='_blank'} can dramatically speed up deep learning tasks and are thus well-suited for our task. They require some specific configuration steps : 

```python
try:
    # TPU detection
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 
    print("Running on TPU ", tpu.cluster_spec().as_dict()["worker"])
    # Connection to TPU
    tf.config.experimental_connect_to_cluster(tpu) 
    # Initialization of the TPU devices
    tf.tpu.experimental.initialize_tpu_system(tpu) 
    # Create a state & distribution policy on the TPU devices
    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except ValueError:
    print("Not connected to a TPU runtime. Using CPU/GPU strategy")
    strategy = tf.distribute.MirroredStrategy()
```
```
Running on TPU  ['10.0.0.2:8470']
```
 According to the [Kaggle TPU documentation](https://www.kaggle.com/docs/tpu){:target='_blank'}, a rule of thumb is to use a batch size of 128 elements per core to take full advantage of the TPU capacities :

```python
BATCH_SIZE = 16 * strategy.num_replicas_in_sync

print('Number of replicas:', strategy.num_replicas_in_sync)
print('Batch size: %.i' % BATCH_SIZE)
```
```
Number of replicas: 8
Batch size: 128
```

## 3. Data import and exploration


```python
# Images
IMG_SOURCE = 640
GCS_DS_PATH = KaggleDatasets().get_gcs_path("resized-plant2021")
TRAIN_PATH = GCS_DS_PATH + f"/img_sz_{IMG_SOURCE}/"
files_ls = tf.io.gfile.glob(TRAIN_PATH + '*.jpg')

# Labels
LABEL_FILE = "../input/plant-pathology-2021-fgvc8/train.csv"
```

```python
mlb = MultiLabelBinarizer()

df_train = pd.read_csv(LABEL_FILE)
df_train["labels_list"] = df_train.labels.apply(lambda x: x.split(' '))

df_class_dummies = pd.DataFrame(mlb.fit_transform(df_train.labels_list),columns=mlb.classes_, index=df_train.index)
df_train = pd.concat([df_train[["image", "labels"]], df_class_dummies], axis=1)

classes_to_predict = mlb.classes_

df_train.head(5)
```
METTRE IMAGE DATAFRAME DE SORTIE

```python
print("Number of examples in the train set : {}".format(len(df_train)))
```
```
Number of examples in the train set : 18632
```
```python
print(classes_to_predict)
```
```
['complex' 'frog_eye_leaf_spot' 'healthy' 'powdery_mildew' 'rust' 'scab']
```



```
AUTO = tf.data.experimental.AUTOTUNE

# Constant variables
IMG_SOURCE = 640
IMG_HEIGHT = 426
IMG_WIDTH = 426





```
