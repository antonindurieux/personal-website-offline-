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

The topic of this article will be to present a solution to geolocate IOT asset trackers thanks to a Machine Learning algorithm.

This project was a group work done during my Post Master in Big Data at Télécom Paris. 

You can find the corresponding python notebook and the data [here](https://github.com/antonindurieux/IoT_geolocalisation) (in french).

## 1. Introduction

The data for this project was provided by [Sigfox](https://www.sigfox.com/en). They consisted of messages sent by IOT asset trackers and received by the Sigfox base stations network in the USA. The goal was to find a way to compute the geolocation of the trackers thanks to machine learning algorithms.

The training set consisted of trackers messages, associated with the position of the trackers at the time of sending. The test set consisted in trackers messages without their position: the task was to compute them. It is important to note that a specific message could have been received by several base stations.

The geolocations are defined by 2 coordinates: the latitude and the longitude. This exercise is thus a **regression** task on these 2 variables.

#### Python imports
```python
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap

from geopy.distance import vincenty

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneGroupOut
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
We have 39250 messages in the training set. For each of these messages, we have the latitude and longitude at the time of emission:

```python
pos_train.head()
```

|    |     lat |      lng |
|---:|--------:|---------:|
|  0 | 39.6067 | -104.958 |
|  1 | 39.6067 | -104.958 |
|  2 | 39.6377 | -104.959 |
|  3 | 39.7304 | -104.969 |
|  4 | 39.7304 | -104.969 |

```python
pos_train.shape
```
```
(39250, 2)
```
There are 29286 messages in the test set:
```python
df_mess_test.shape
```
```
(29286, 8)
```

Some interesting basic information would be:
- The number of base stations;
- The number of different messages in the training and test set;
- The number of different devices in the training and test set.

```python
listOfBs = np.union1d(np.unique(df_mess_train['bsid']), np.unique(df_mess_test['bsid'])) 
print("Total number of base stations : {}".format(len(listOfBs)))

listOfMessId_train = np.unique(df_mess_train['messid'])
print("Number of different messages in the training set : {}".format(len(listOfMessId_train)))

listOfMessId_train = np.unique(df_mess_train['messid'])
print("Number of different messages in the training set : {}".format(len(listOfMessId_train)))

listOfMessId_train = np.unique(df_mess_train['messid'])
print("Number of different messages in the test set : {}".format(len(listOfMessId_train)))

listOfMessId_train = np.unique(df_mess_train['did'])
print("Number of different devices in the training set : {}".format(len(listOfMessId_train)))

listOfMessId_train = np.unique(df_mess_train['did'])
print("Number of different devices in the test set : {}".format(len(listOfMessId_train)))
```
```
Total number of base stations : 259
Number of different messages in the training set : 6068
Number of different messages in the test set : 5294
Number of different devices in the training set : 113
Number of different devices in the test set : 56
```

## 3. Positionning

To understand the problem, it seems important to check where the devices of the training set and the base stations are located: 

```python
# Background
fig = plt.figure(figsize=(16, 16))
m = Basemap(projection='cyl', resolution='l',
            llcrnrlon=-115, llcrnrlat= 35, 
            urcrnrlon=-65, urcrnrlat=65)
m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')
m.drawparallels(np.arange(35,65,5), labels=[1,1,1,1])
m.drawmeridians(np.arange(-110,-65,5), labels=[1,1,1,1])

# Test base stations 
bs_lat_test, bs_lng_test = m(df_mess_test.bs_lat.values, df_mess_test.bs_lng.values)
plt.plot(bs_lng_test, bs_lat_test, marker='^', color='m', markersize=4, linestyle="None", label='Test base stations')

# Training base stations
bs_lat_train, bs_lng_train = m(df_mess_train.bs_lat.values, df_mess_train.bs_lng.values)
plt.plot(bs_lng_train, bs_lat_train, '^b', markersize=4, label='Train base stations')

# Devices
device_lat, device_lng = m(pos_train.lat.values, pos_train.lng.values)
plt.plot(device_lng, device_lat, 'xg', markersize=2, label='Devices')

# Outliers
circle = plt.Circle((-68.5, 64.3), 1, color='r', fill=False, lw=4)
plt.gca().add_patch(circle)

plt.legend()
plt.show()
```
![img1](/assets/images/iot_img1.png)

We see that most of the devices and base stations are located on the east flank of the rocky mountains in the USA, in a square between 110°W and 100°W / 35°N and 45°N.

There are obvious outliers among the base stations (circled in red on the map): some of them are located far to the north-east, in the Nunavut (Canada). It is probably impossible that they could have receive messages that far from the devices, their position thus must be erroneous.

```python
print("Base stations located in the Nunavut - training set : ")
print(np.unique(df_mess_train[df_mess_train.bs_lng > -70]['bsid']))
```
```
Base stations located in the Nunavut - training set : 
[ 1092  1594  1661  1743  1772  1796  1854  2293  2707  2943  4123  4129
  4156  4959  4987  4993  7248  8355  8449  8451  8560  9784 10151 10162
 10999 11007 11951]
 ```
 ```python
print("Base stations located in the Nunavut - test set : ")
print(np.unique(df_mess_test[df_mess_test.bs_lng > -70]['bsid']))
```
```
Base stations located in the Nunavut - test set : 
[ 1092  1594  1661  1743  1772  1796  1854  2707  2943  4129  4156  4987
  4993  7248  8355  8449  8451  8560  9941  9949 10151 10162 11007]
 ```

There are 27 stations that seem wrong-located in the training set and 23 in the test set.

If we zoom on the south-western corner of the map, here is what we get:
![img2](/assets/images/iot_img2.png)

The stations located around 106°W / 44°N are also suspicious.

We will see later if we should keep the data from the wrongly located stations or if we should handle them in any specific way.

## 4. Computation of the feature matrix
The idea of our algorithm will be to compute a feature matrix $${X}$$ where:
- each row corresponds to a message;
- each column corresponds to a base station;
- aaaaaa
