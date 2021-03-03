---
title: "Unsupervised anomaly detection on accelerometer data"
excerpt: "My solution to an unsupervised anomaly detection data challenge, on accelerometer time-series measurements acquired during helicopter flights."
header:
  overlay_image: /assets/images/helicopter.jpg
  show_overlay_excerpt: false
  caption: <span>Photo by <a href="https://unsplash.com/@spacedezert?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">SPACEDEZERT</a> on <a href="https://unsplash.com/s/photos/helicopters?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>
  teaser: assets/images/helicopter.jpg
classes: wide
---
I will present here my solution for an unsupervised anomaly detection task on accelerometer data that were acquired during helicopter flights.

This exercice was presented as a data challenge competition during my Post Master in Big Data at Télécom Paris. I was very motivated for this challenge since anomaly detection is one of the machine learning applications that I find the most interesting. I ranked 2nd out of 51 participants.

You can find the full python notebook [here](https://github.com/antonindurieux/data_challenge-unsupervised_anomaly_detection/blob/master/Data_challenge-Detection_anomalies_non_supervisee.ipynb) (in french).

# 1. Introduction
## 1.1 DTask and data

The data for this challenge was provided by Airbus. They consisted of  time-series of accelerometer data, that were acquired during helicopter flights. Each observation was a 1 minute recording, sampled at 1024 Hz (which thus makes 61440 samples by records).  

The training set consisted of 1677 of those 1-minute recordings, while de test set consisted of 2511. The task was to affect an anomaly score to each test recording : the higher the score, the more abnormal the recording should be. No label was provided, which made it an **unsupervised** anomaly detection task.

The performance criterion was the [**Area Under the Receiver Operating Characteristic (AUC)**](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve).

One of the main challenge was that **no hints or indication was given about what constitued an "abnormal" time-serie**. Hence, anomalies had to be identified in an automatic way by learning the normal behavior, that of the vast majority of the observations, and considering those differing significantly from it as abnormal. Logically, anomalies are rare in the data and thus fall in low density regions: anomaly detection thus boils down to identifying the tail of the distribution.

## 1.2 My approach
As I was very interested in this task, I took the time to try various approaches to see what looked promising. The method which gave me the best submission score consisted in extracting statistical features from the time-series, and add some frequency-domain information from their periodograms. I then used the [one-class SVM](https://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html) algorithm which can be used to detect anomalies or novelties.


### Python imports
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from scipy import signal
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

sns.set()
```

# 2. Data import
As the dataset could take some time to be loaded from their raw csv format, I previously converted them as [npy](https://numpy.org/doc/stable/reference/generated/numpy.save.html) for faster loading.
```python
xtrain = np.empty((1677,61440))
xtest = np.empty((2511,61440))

xtrain = np.load(file='drive/My Drive/Data_Challenge_MDI341/Raw_Data/xtrain_raw_array.npy')
xtest = np.load(file='drive/My Drive/Data_Challenge_MDI341/Raw_Data/xtest_raw_array.npy')
```


# Statistical features extraction

WIP

# Extraction of frequency information

# Anomaly scores calculation

WIP

# Conclusion

WIP