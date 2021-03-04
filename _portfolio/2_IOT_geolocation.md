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

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>


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
The idea of our algorithm will be to compute a feature matrix $$X$$ where:
- each row $$i$$ corresponds to a message;
- each column $$j$$ corresponds to a base station;
- $$X_{ij} = RSSI_{ij}$$ if the message $$i$$ has been received by the base station $$j$$, $$X_{ij} = 0$$ otherwise.

We will subsequently use this feature matrix to feed a k-nearest neighbors regressor algorithm. This algorithm will allow to interpolate locations based on the positionning of the neighboring devices, taking the RSSI values as weights.

From this choice of feature matrix, we consider that we can keep the wrongly located stations in our training set. Indeed, our solution is independent of the coordinates of the base stations. What matters is that the training devices are correctly located, and that we don't have abnormal RSSI values.

```python
sns.set()

plt.figure(figsize=(16, 5))
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

df_mess_train.hist(column='rssi', bins=50, ax=ax1, density=1)
ax1.set_title("RSSI - Training data")
ax1.set_xlabel("RSSI")
ax1.set_ylabel("Densité")

df_mess_test.hist(column='rssi', bins=50, ax=ax2, density=1, color='orange')
ax2.set_title("RSSI - Test data")
ax2.set_xlabel("RSSI")
ax2.set_ylabel("Densité")

plt.show()
```
![img3](/assets/images/iot_img3.png)

We can see that the RSSI values are similarly distributed between the training and the test set. We don't spot any outlier.  

Here is the function to generate our feature matrix:

```python
def feat_mat_const(df_mess_train, listOfBs):
    """
    Computation of the feature matrix
    Input: 
        df_mess_train - DataFrame of messages
        listOfBs - list of base station ids
    Output:
        df_feat - feature matrix DataFrame
        id_list - ordered list of message ids
    """
    
    df_mess_bs_group = df_mess_train.groupby(['messid'], as_index=False) # Group data by message (messid)
    nb_mess = len(np.unique(df_mess_train['messid']))
    df_feat = pd.DataFrame(np.zeros((nb_mess,len(listOfBs))), columns = listOfBs) # Feature matrix initialization
    df_feat['did'] = np.nan # Device id column initialization
    
    idx = 0
    id_list = [0] * nb_mess

    for key, elmt in df_mess_bs_group:      
        # Filling the matrix with RSSI
        for bsid in df_mess_bs_group.get_group(key)['bsid']:
            rssi = df_mess_train.loc[(df_mess_train.messid==key) & (df_mess_train.bsid==bsid)]['rssi'].values
            df_feat.loc[idx,bsid] = rssi
        # Filling the device id column
        if df_mess_bs_group.get_group(key).did.nunique() != 1:
            print("Error: non-unique device id for message {}".format(key))
        else:
            device_id = df_mess_bs_group.get_group(key).did.values[0]
        df_feat.loc[idx,'did'] = device_id
        
        id_list[idx] = key
        idx = idx + 1
    
    return df_feat, id_list
```

We will also have to compute the ground truth associated with each distinct message:

```python
def ground_truth_const(df_mess_train, pos_train):
    """
    Computation of ground truth latitudes and longitudes ordered lists 
    Input: 
        df_mess_train - DataFrame of messages
        pos_train - DataFrame of locations
    Output:
        ground_truth_lat - array of ground truth latitudes
        ground_truth_lng - array of ground truth longitudes
    """

    df_mess_pos = df_mess_train.copy()
    df_mess_pos[['lat', 'lng']] = pos_train

    ground_truth_lat = np.array(df_mess_pos.groupby(['messid']).mean()['lat'])
    ground_truth_lng = np.array(df_mess_pos.groupby(['messid']).mean()['lng'])
    
    return ground_truth_lat, ground_truth_lng
```
```python
# Feature matrix computation
df_feat, id_list_train = feat_mat_const(df_mess_train, listOfBs)

# Ground truth of each message
ground_truth_lat, ground_truth_lng = ground_truth_const(df_mess_train, pos_train)
ground_truth = np.stack([ground_truth_lat, ground_truth_lng], axis=1)
```
```python
# Feature matrix
df_feat.head()
```

|    |   879 |   911 |   921 |   944 |   980 |   1012 |   1086 |   1092 |   1120 |   1131 |   1148 |   1156 |   1187 |   1226 |   1229 |   1235 |   1237 |   1264 |   1266 |   1268 |   1292 |   1334 |   1344 |   1432 |   1443 |   1447 |   1463 |   1476 |   1526 |   1530 |   1534 |   1581 |   1594 |   1661 |   1730 |     1741 |   1743 |   1772 |   1796 |   1826 |   1828 |   1838 |   1852 |   1854 |     1859 |   1872 |     1878 |   1971 |   1987 |   1988 |   1994 |   1996 |   2189 |   2293 |     2605 |     2610 |   2611 |   2617 |   2693 |   2707 |   2731 |   2737 |   2762 |   2765 |   2766 |   2768 |   2770 |   2775 |   2776 |   2780 |   2784 |   2790 |   2799 |   2800 |   2803 |   2808 |   2831 |   2836 |   2837 |   2841 |   2842 |   2845 |   2846 |   2849 |   2855 |   2862 |   2943 |   2945 |   2999 |     3025 |   3034 |   3041 |   3051 |     3256 |   3357 |   3378 |   3385 |   3386 |   3389 |   3402 |   3403 |   3410 |   3412 |   3414 |   3415 |   3500 |   3501 |   3515 |   3526 |   3527 |     3529 |   3535 |   3536 |   3538 |   3544 |   3545 |   3546 |   3547 |   3548 |   3549 |   3553 |   3554 |   3555 |     3556 |   3558 |   3559 |     3562 |   3563 |   3565 |   3568 |   3569 |   3570 |   3571 |   3572 |   3574 |   3575 |   3576 |   3577 |   3578 |   3579 |     3581 |   3613 |   3629 |   3630 |   3646 |   3828 |   3835 |   3846 |   3848 |   3907 |   3915 |   3933 |   3983 |   4013 |   4024 |   4047 |   4049 |   4055 |   4056 |   4058 |   4059 |   4060 |   4064 |   4065 |   4073 |   4078 |   4088 |   4092 |   4105 |   4123 |   4129 |   4147 |   4148 |   4156 |   4157 |   4205 |   4244 |   4646 |   4790 |   4813 |   4819 |   4959 |   4966 |   4987 |   4993 |   4996 |   7248 |   7378 |   7382 |   7435 |   7456 |   7490 |   7508 |   7628 |   7655 |   7673 |   7692 |   7726 |   7738 |   7789 |   7807 |   7849 |   7857 |   7972 |   7986 |   8079 |   8082 |   8168 |   8226 |   8242 |   8245 |   8351 |     8352 |   8355 |     8356 |   8364 |   8368 |   8370 |   8371 |   8384 |   8390 |   8392 |   8397 |   8401 |   8405 |   8426 |   8437 |   8442 |   8446 |   8449 |   8450 |   8451 |   8452 |   8453 |   8457 |   8470 |   8471 |   8472 |   8473 |   8474 |   8475 |   8495 |   8509 |   8560 |   8746 |   8747 |   9783 |   9784 |   9899 |   9936 |   9941 |   9949 |   10134 |   10148 |   10151 |   10162 |   10999 |   11007 |   11951 |    did |
|---:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|---------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|---------:|-------:|---------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|---------:|---------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|---------:|-------:|-------:|-------:|---------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|---------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|---------:|-------:|-------:|---------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|---------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|---------:|-------:|---------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|-------:|
|  0 |     0 |     0 |     0 |     0 |     0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |    0     |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 | -121.5 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |      0 |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |   -125 |      0 |    0     |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 | 473335 |
|  1 |     0 |     0 |     0 |     0 |     0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |      0 | -134     |    0     |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |      0 |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 | 473335 |
|  2 |     0 |     0 |     0 |     0 |     0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     | -132     |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |      0 |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |      0 |   -120 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |   -100 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 | 473953 |
|  3 |     0 |     0 |     0 |     0 |     0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 | -123.333 |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |      0 | -129.667 |    0     |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 | -123.667 |      0 |      0 |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |   -133 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |    0     |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 | 476512 |
|  4 |     0 |     0 |     0 |     0 |     0 |      0 |   -141 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 | -116.667 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 | -120.333 |      0 |   -138 |   -138 |      0 |   -131 |   -120 |      0 | -132     | -125.667 |      0 | -137.5 |      0 |      0 |   -108 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 | -125.5 |      0 |   -117 |      0 |      0 |      0 |      0 |      0 |      0 |    0   |      0 |      0 |   -130 |      0 |      0 |      0 |      0 |      0 |      0 |    0     |      0 |      0 |      0 | -136.333 |      0 |      0 |      0 |      0 |      0 |      0 |      0 | -121.5 |   -123 |      0 |      0 |      0 |      0 |      0 |   -136 |      0 | -108.667 |      0 |      0 | -135.5 |      0 |      0 |      0 |   -116 |   -128 |      0 |      0 |      0 |   -125 | -115.667 |   -131 |      0 | -114.333 |      0 |   -134 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |    -98 | -122.667 |      0 |      0 |      0 |      0 |      0 |      0 | -125.5 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |   -114 |      0 |   -124 |      0 |      0 |      0 |      0 | -139.5 |      0 |      0 |      0 |      0 |      0 |   -132 |      0 |      0 |   -133 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 | -106.667 |      0 | -123.333 |      0 |      0 |      0 |      0 |      0 |      0 |      0 | -127.5 |      0 |      0 |   -139 | -137.5 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 | -134.5 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |   -129 |      0 |      0 |      0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 | 476286 |

```python
# Feature matrix dimensions
df_feat.shape
```
```
(6068, 260)
```

```python
print(ground_truth)
```
```
[[  39.60668952 -104.95848993]
 [  39.63774123 -104.95855416]
 [  39.73041743 -104.96894015]
 ...
 [  39.77887196 -105.01928515]
 [  39.77326419 -105.01405242]
 [  39.90818599 -105.16829706]]
```

```python
ground_truth.shape
```
```
(6068, 2)
```

## 5. Leave-one-device-out cross-validation and modeling

To select the best hyper-parameters of the k-Nearest Neighbors algorithm, we will use a leave-one-device-out cross-validation: every device is put aside one after the other during the successive trainings, to get a prediction on the corresponding never-seen device. The total resulting error will thus be calculated by taking into account each one of the devices. This method eliminate the bias that some may be easier to locate than others.

For the k-Nearest Neighbors, the important hyper-parameter is the number of neighbors taken into account, $$k$$. The optimal $$k$$ setting will be defined thanks to this cross-validation procedure. As the predictions on latitudes and longitudes are separate, we can train a k-Nearest Neighbors for each coordinate.

### 5.1 Error evaluation
To check how good are the results, we will compute the [Vincenty distance](https://en.wikipedia.org/wiki/Vincenty%27s_formulae) between the ground truth and the predicted location, in meters:
```python
def vincenty_vec(vec_coord):
    """
    Computation of the Vincenty distances between pairs of points
    Input: 
        vec_coord - array of coordinates with column [y_lat , y_lng, y_pred_lat, y_pred_lng]
    Output:
        vin_vec_dist - array of Vincenty distances in meters
    """
    
    vin_vec_dist = np.zeros(vec_coord.shape[0])
    if vec_coord.shape[1] !=  4:
        print('ERROR: Bad number of columns (shall be = 4)')
    else:
        vin_vec_dist = [vincenty(vec_coord[m,0:2],vec_coord[m,2:]).meters for m in range(vec_coord.shape[0])]
    return vin_vec_dist
```
 WIP

```python
# We remove the device id from the feature matrix
features = df_feat.columns.values[:-1]
```