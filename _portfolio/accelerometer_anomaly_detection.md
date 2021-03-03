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
## 1.1 Task and data

The data for this challenge was provided by Airbus. They consisted of  time-series of accelerometer data, that were acquired during helicopter flights. Each observation was a 1 minute recording, sampled at 1024 Hz (which thus makes 61440 samples by records).  

The training set consisted of 1677 of those 1-minute recordings, while de test set consisted of 2511. The task was to affect an anomaly score to each test recording : the higher the score, the more abnormal the recording should be. No label was provided, which made it an **unsupervised** anomaly detection task.

The performance criterion was the [**Area Under the Receiver Operating Characteristic (AUC)**](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve).

One of the main challenge was that **no hints or indication was given about what constitued an "abnormal" time-serie**. Hence, anomalies had to be identified in an automatic way by learning the normal behavior, that of the vast majority of the observations, and considering those differing significantly from it as abnormal. Logically, anomalies are rare in the data and thus fall in low density regions: anomaly detection thus boils down to identifying the tail of the distribution.

Another challenge was that as it was an unsupervised task, the only available performance feedback was the score calculated by the submission website.

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

print(xtrain.shape)
print(xtest.shape)
```
```
(1677, 61440)
(2511, 61440)
```
By visualizing the time-series, it is not very clear on which specific criteria a recording could be considered abnormal. On the other hand, by randomly checking them, it appears that some of them are clearly exhibiting abnormal features :

```python
# Exemples of series
time_vect = np.array(range(61440)) / 1024

plt.figure(figsize=(20,4))

plt.subplot(1, 3, 1)
plt.plot(time_vect, xtrain[1])
plt.title("Probably normal serie")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration")

plt.subplot(1, 3, 2)
plt.plot(time_vect, xtest[1794])
plt.title("Saturated serie, probably abnormal")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration")

plt.subplot(1, 3, 3)
plt.plot(time_vect, xtest[2368])
plt.title("Flat serie, probably abnormal")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration")

plt.show()
```

![img1](/assets/images/anomaly_detection_img1.png)
(The acceleration unit was not specified, it could be m/s<sup>2</sup>, Gal, g, or raw volts).

# Statistical features extraction

I used the [tsfresh](https://tsfresh.readthedocs.io/en/latest/) Python package to automatically calculates a large number of time series characteristics on each recording. 

As I didn't have information on which features could make sense to characterize abnormal series, I chose to extract a large set of them that seemed of possible use. The process of feature calculation taking a lot of time, I [pickled](https://docs.python.org/3/library/pickle.html) the output DataFrames.

```python
file = '/content/drive/My Drive/Data_Challenge_MDI341/Pickles/xtrain_stat_feats_df'
xtrain_feats_df = pd.read_pickle(file)

file = '/content/drive/My Drive/Data_Challenge_MDI341/Pickles/xtest_stat_feats_df'
xtest_feats_df = pd.read_pickle(file)

xtrain_feats_df.head()
```

|   id |   ts__abs_energy |   ts__absolute_sum_of_changes |   ts__count_above_mean |   ts__count_below_mean |   ts__has_duplicate_max |   ts__has_duplicate_min |   ts__kurtosis |   ts__maximum |   ts__mean |   ts__mean_abs_change |   ts__mean_second_derivative_central |   ts__median |   ts__minimum |   ts__skewness |   ts__standard_deviation |   ts__sum_values |   ts__variance |
|-----:|-----------------:|------------------------------:|-----------------------:|-----------------------:|------------------------:|------------------------:|---------------:|--------------:|-----------:|----------------------:|-------------------------------------:|-------------:|--------------:|---------------:|-------------------------:|-----------------:|---------------:|
|    1 |          133.394 |                       851.568 |                  30749 |                  30691 |                       0 |                       0 |      13.5778   |      0.299668 | 0.0393726  |             0.0138604 |                         -3.00433e-07 |    0.040173  |     -0.247552 |    -0.0400113  |                0.0249183 |         2419.05  |    0.000620922 |
|    2 |        29724.1   |                     12468.6   |                  30584 |                  30856 |                       0 |                       0 |      -1.40717  |      1.42342  | 0.00372549 |             0.202943  |                          1.58168e-06 |   -0.0027145 |     -1.2953   |     0.0546057  |                0.695541  |          228.894 |    0.483777    |
|    3 |        44641.6   |                     15004.2   |                  31060 |                  30380 |                       0 |                       0 |      -1.31412  |      1.74834  | 0.0127751  |             0.244212  |                         -3.46738e-06 |    0.029049  |     -1.74511  |     0.00221549 |                0.852306  |          784.902 |    0.726425    |
|    4 |         1294.07  |                      5864.89  |                  30710 |                  30730 |                       0 |                       1 |       0.574434 |      0.71788  | 0.0049188  |             0.0954587 |                          9.60505e-07 |    0.004331  |     -0.748198 |    -0.0108893  |                0.145045  |          302.211 |    0.0210381   |
|    5 |        26504.8   |                     11783.1   |                  30279 |                  31161 |                       0 |                       0 |      -1.41685  |      1.40279  | 0.0322576  |             0.191785  |                          8.12925e-07 |    0.008686  |     -1.21387  |     0.118328   |                0.656013  |         1981.91  |    0.430353    |

In these DataFrames, each row corresponds to one time-serie and each column to a feature.

```python
# List of extracted features
print(xtrain_feats_df.columns.values)
```
```
['ts__abs_energy' 'ts__absolute_sum_of_changes' 'ts__count_above_mean'
 'ts__count_below_mean' 'ts__has_duplicate_max' 'ts__has_duplicate_min'
 'ts__kurtosis' 'ts__maximum' 'ts__mean' 'ts__mean_abs_change'
 'ts__mean_second_derivative_central' 'ts__median' 'ts__minimum'
 'ts__skewness' 'ts__standard_deviation' 'ts__sum_values' 'ts__variance']
```

At this stage, **a crucial step has been to select the useful features** of this set. By selecting a small subset of them, then adding/deleting one feature at a time, [and feeding the data into my one-class SVM](#anomaly-scores-calculation), I could then submit my results and see in which direction my score evolved after submitting.

This process led me to a very surprising result. **From these 17 statistical features, only 3 positively contributed to the score, and these were among the most basic**: the mean, the standard deviation and the median.

Any other statistical feature added would significantly decrease the score. They likely didn't contain any additional useful information in this context, and introduced some noise. 

```python
# Statistical features selection
feats = ['ts__mean',
         'ts__standard_deviation',
         'ts__median'
         ]

xtrain_feats_df = xtrain_feats_df[feats]
xtest_feats_df = xtest_feats_df[feats]
```
```python
# Distribution of the features in the train and test sets
plt.figure(figsize=(20,4))

for i, col in enumerate(xtrain_feats_df.columns): 
    plt.subplot(1, 3, i+1)
    bins = np.histogram(np.hstack((xtrain_feats_df[col],xtest_feats_df[col])), bins=20)[1]
    plt.hist(xtrain_feats_df[col], alpha=0.5, bins=bins, density=True, label='Train')
    plt.hist(xtest_feats_df[col], alpha=0.5, bins=bins, density=True, label='Test')
    plt.yscale('log')
    plt.title(col)
    plt.legend()
```
![img2](/assets/images/anomaly_detection_img2.png)

We see that some parts of the feature distributions are filled only by the test set. This was promising: these differences could be explained by abnormal series.

By applying a one-class SVM only on those 3 features, the AUC score was already equal to 0.836. 

# Extraction of frequency information

The next step consisted in calculating periodograms of the signals to see if some interesting frequency information could be added.

```python
# Periodograms calculation
f, Pxx_den_train = signal.welch(xtrain, 
                          fs=1024, 
                          window='hanning', 
                          nperseg=256,
                          noverlap=None, 
                          nfft=None,
                          return_onesided=True, 
                          scaling='spectrum', 
                          axis=-1)

f, Pxx_den_test = signal.welch(xtest, 
                          fs=1024, 
                          window='hanning',
                          nperseg=256, 
                          noverlap=None, 
                          nfft=None, 
                          return_onesided=True, 
                          scaling='spectrum', 
                          axis=-1)
```

I wanted to check if we could spot differences in the periodograms between the series rated the most normal and the most abnormal so far (i.e. after applying the one-class SVM to the 3 statistical features).

```python
high_scores = [2362, 2363, 2364, 1983, 1980, 2368, 2369, 2356, 2370, 2378, 2379,
       1972, 2383, 1971, 1970, 2386, 2387, 1968, 2391, 2392, 2372, 2071,
       2355, 2353, 2062, 2060, 2311, 2051, 2313, 2048,  654, 2042, 2040,
       2320, 2321, 1997, 2325, 2328, 2033, 2032, 2337,  225, 2018, 2344,
       2007, 2348, 1999, 1998, 2037, 2002]

low_scores = [1578, 1248, 1721,  318, 1397, 1350,  185,  242,   69, 1566, 1252,
        609,  666, 1104, 1531, 1170,   96,  465, 1249, 1459,  671,  875,
       1699, 2441,  386,  712,  631, 1229,  491, 1627,  385, 1750, 1496,
       1483, 1489, 1884,  618,  432,  894,  238, 1402, 1672,  299,  547,
       1421, 1475,  809,  648, 2184, 1213]

plt.figure(figsize=(25, 7))

plt.subplot(1, 2, 1)
plt.plot(f, Pxx_den_test[low_scores].T)
plt.title("Frequency spectrum of the a priori most normal series")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power spectral density")

plt.subplot(1, 2, 2)
plt.plot(f, Pxx_den_test[high_scores].T)
plt.title("Frequency spectrum of the a priori most abnormal series")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power spectral density")

plt.show()
```

![img3](/assets/images/anomaly_detection_img3.png)

We can see that the signal frequencies repartition and pics can be very different and that the normal series seem more homogeneous. 

I thus tried different feature engineering options to exploit these differences in the power spectrums. The solution that I found working the best was to compute :
- The frequency of the 25<sup>th</sup> percentile of the total power spectrum of each serie;
- The frequency of the 50<sup>th</sup> percentile of the total power spectrum of each serie;

```python
def f_cumsum(p_vect, f_vect, percentile):
    """
    Compute the frequency corresponding to a percentile of a power spectrum
    Input: 
        p_vect - array of powers
        f_vect - array of frequencies
        percentile - percentile of the spectrum for which to compute the frequency
      Output:
        f - frequency corresponding to the percentile
        p_sum - cumulative sum of power at the percentile
    """
    cumsum = np.cumsum(p_vect)
    break_point = cumsum[-1] * percentile
    arg = np.argmax(cumsum >= break_point) + 1
    f = f_vect[arg]
    p_sum = np.sum(p_vect[:arg])

    return f, p_sum
```
```python
# Computation of the frequencies of the 25th and 50th percentile of each serie
train_f_p025 = []
train_f_p05 = []
test_f_p025 = []
test_f_p05 = []

for i in range(len(Pxx_den_train)):
    p_vect = Pxx_den_train[i]
    train_f_p025.append(f_cumsum(p_vect, f, 0.25)[0]) 
    train_f_p05.append(f_cumsum(p_vect, f, 0.5)[0]) 

for i in range(len(Pxx_den_test)):
    p_vect = Pxx_den_test[i]
    test_f_p025.append(f_cumsum(p_vect, f, 0.25)[0]) 
    test_f_p05.append(f_cumsum(p_vect, f, 0.5)[0]) 

xtrain_feats_df['f_p025'] = train_f_p025
xtrain_feats_df['f_p05'] = train_f_p05
xtest_feats_df['f_p025'] = test_f_p025
xtest_feats_df['f_p05'] = test_f_p05
```

We can plot the repartition of those 2 new features as histograms and a scatterplot :

```python
plt.figure(figsize=(20,8))

for i, col in enumerate(xtrain_feats_df.columns[-2:]): 
    plt.subplot(2, 3, i+1)
    bins = np.histogram(np.hstack((xtrain_feats_df[col],xtest_feats_df[col])), bins=20)[1]
    plt.hist(xtrain_feats_df[col], alpha=0.5, bins=bins, density=True, label='Train')
    plt.hist(xtest_feats_df[col], alpha=0.5, bins=bins, density=True, label='Test')
    plt.title(col)
    plt.xlabel('Frequency (Hz)')
    plt.yscale('log')
    plt.legend()
```

![img4](/assets/images/anomaly_detection_img4.png)

# Anomaly scores calculation

WIP

# Conclusion

WIP