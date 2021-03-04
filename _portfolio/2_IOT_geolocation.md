---
title: "IOT trackers geolocation"
excerpt: "Development of a machine learning solution to geolocate IOT asset trackers."
header:
  overlay_image: /assets/images/map.jpg
  show_overlay_excerpt: false
  caption: <span>Photo by <a href="https://unsplash.com/@timowielink?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Timo Wielink</a> on <a href="https://unsplash.com/s/photos/map?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>
  teaser: assets/images/map.jpg
classes: wide
---

The topic of this article will be to present a solution to geolocate IOT asset trackers thanks to Machine Learning techniques.

This project was a group work done during my Post Master in Big Data at Télécom Paris. 

You can find the corresponding python notebook and the data [here](https://github.com/antonindurieux/IoT_geolocalisation) (in french).

## 1. Introduction

The data for this project was provided by [Sigfox](https://www.sigfox.com/en). They consisted of messages sent by IOT asset trackers and received by the Sigfox base stations network in the USA. The goal was to find a way to compute the geolocation of the trackers thanks to machine learning algorithms.

The training set consisted of the trackers messages, associated with their position (the ground truth). The test set consisted just in the trackers messages without their position, the task was to compute them.

The geolocations are defined by 2 coordinates: the latitude and the longitude. This exercise is thus a **regression** task on these 2 variables.

#### Python imports
```python
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import vincenty
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneGroupOut
import seaborn as sns
```

## 2. Data import
```python
df_mess_train = pd.read_csv('mess_train_list.csv')
df_mess_test = pd.read_csv('mess_test_list.csv')
pos_train = pd.read_csv('pos_train_list.csv')
```
```python
df_mess_train.head()
```

|    | messid                   |   bsid |    did |   nseq |   rssi |     time_ux |   bs_lat |   bs_lng |
|---:|:-------------------------|-------:|-------:|-------:|-------:|------------:|---------:|---------:|
|  0 | 573bf1d9864fce1a9af8c5c9 |   2841 | 473335 |    0.5 | -121.5 | 1.46355e+12 |  39.6178 | -104.955 |
|  1 | 573bf1d9864fce1a9af8c5c9 |   3526 | 473335 |    2   | -125   | 1.46355e+12 |  39.6773 | -104.953 |
|  2 | 573bf3533e952e19126b256a |   2605 | 473335 |    1   | -134   | 1.46355e+12 |  39.6127 | -105.009 |
|  3 | 573c0cd0f0fe6e735a699b93 |   2610 | 473953 |    2   | -132   | 1.46355e+12 |  39.798  | -105.073 |
|  4 | 573c0cd0f0fe6e735a699b93 |   3574 | 473953 |    1   | -120   | 1.46355e+12 |  39.7232 | -104.956 |

The information contained in each messages are the following:
- **messid** is the message id;
- **bsid** is the reception base station id;
- **did** is the device (tracker) id;
- **nseq** is a variable whose meaning was not provided, maybe it could be usefull, maybe not...
- **rssi** is the [received signal strength](https://fr.wikipedia.org/wiki/Received_Signal_Strength_Indication);
- **time_ux** is the reception time;
- **bs_lat** and **bs_lng** are the coordinates of the reception base station.

```python
df_mess_train.head()
```
```
(39250, 8)
```
We have 39250 messages in the training set.

