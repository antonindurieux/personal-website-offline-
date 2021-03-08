---
title: "Airbnb price modeling with Spark"
excerpt: "How to use Spark to work on the Paris Airbnb dataset, from data cleaning to price modeling."
header:
  overlay_image: /assets/images/paris.jpg
  show_overlay_excerpt: true
  caption: <span>Photo by <a href="https://unsplash.com/@nicolasjehly?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Nicolas Jehly</a> on <a href="https://unsplash.com/s/photos/paris?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>
  teaser: assets/images/paris.jpg
classes: wide
---

**Work In Progress**

On this article I will show how to use [Spark](https://spark.apache.org/){:target="_blank"} to work on the Paris Airbnb listings. I will put myself in the shoes of someone who put his parisian property on the famous lodging platform. A good question would thus be: given the features of my apartment (location, accomodation capacity, bedrooms...), as well as the flexibility of my booking rules (minimum and maximum number of nights, instant booking, cancellation rules...), and how much guests appreciate my accomodation, what would be a price by night aligned with the competition?

This scenario will be a good excuse to:
- Show how to work with [PySpark](https://spark.apache.org/docs/latest/api/python/){:target="_blank"};
- Clean the dataset and explore some aspects of it;
- Produce a price model.

The iPython notebook of this project (with some adaptations) can be found [here](https://github.com/antonindurieux/Airbnb-price-analysis-with-Spark) (in french).

## 1. The data

The data will be downloaded from http://insideairbnb.com/. As specified on the website, "Inside Airbnb is an independent, non-commercial set of tools and data that allows you to explore how Airbnb is really being used in cities around the world". There is a lot of very interesting and valuable data on this website, but we must keep in mind that the data is not from "official sources" in case of inconsistencies.

I will use the Paris data, and more specifically the listings csv file which gives the characteristics of every listing in the city.

## 2. From the raw data to a Spark DataFrame

I will start by downloading and de-zipping the data. I will then show how to load it in a RDD then to a DataFrame. An RDD is a [Resilient Distributed Datasets](https://spark.apache.org/docs/3.1.1/rdd-programming-guide.html#resilient-distributed-datasets-rdds){:target="_blank"}, the fundamental data structure of Spark. A SPark DataFrame ["is conceptually equivalent to a table in a relational database or a data frame in R/Python, but with richer optimizations under the hood"](https://spark.apache.org/docs/latest/sql-programming-guide.html){:target="_blank"}. At the time of this writing, Python still did not support another Spark data structure, the [Datasets](https://spark.apache.org/docs/latest/sql-programming-guide.html){:target="_blank"}.

##### Python imports
```python
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

import matplotlib.pyplot as plt
import numpy as np
import csv
import pprint
import seaborn as sns
import operator
import pandas as pd
import geopandas as gpd
import requests
import gzip
import shutil
```

### 2.1 Data import

```python
url = "http://data.insideairbnb.com/france/ile-de-france/paris/2019-09-16/data/listings.csv.gz"
filename = url.split("/")[-1]
with open(filename, "wb") as f:
    r = requests.get(url)
    f.write(r.content)

with gzip.open('listings.csv.gz', 'rb') as f_in:
    with open('data/listings.csv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

listings_csv = "data/listings.csv"
```

The first cleaning step is to remove the newline characters contained in some of the fields of the data. Otherwise the whole data structure will be messed up and there will be errors while loading it.

```python
with open(listings_csv, "r") as input, open("data/cleaned_listings.csv", "w") as output:
    w = csv.writer(output)
    for record in csv.reader(input):
        w.writerow(tuple(s.replace('\n', '') for s in record))
output.close()
```

### 2.2 Creating an RDD from the data

Some functions will be required to cast the different fields while handling missing values: 

```python
def parseCSV(file):
    """
    Return a reader from a csv
    """
    reader = csv.reader(file.splitlines(), skipinitialspace=True) 
    return(next(reader)) 

def toFloat(field): 
    """
    Cast the input to a float
    """
    if field == '':
        return None
    else:
        return float(field) 

def toInt(field):
    """
    Cast the input to an integer
    """
    if field == '':
        return None
    else:
        return int(field)

def toString(field):
    """
    Cast the input to a string
    """
    if field == '':
        return None
    else:
        return str(field)
```

It is time to launch a [Spark context](https://spark.apache.org/docs/2.3.0/api/java/org/apache/spark/SparkContext.html){:target="_blank"} and a [Spark session](https://spark.apache.org/docs/2.2.1/api/python/pyspark.sql.html?highlight=sqlcontext#pyspark.sql.SparkSession):

```python
sc = SparkContext()
sparkSession = SparkSession(sc)
```

And now we can create an RDD from the data:

```python
listings_rdd = sc.textFile('data/cleaned_listings.csv')

# Remove empty lines
listings_rdd = listings_rdd.filter(lambda line: line != '') 

# Parse the data
listings_rdd = listings_rdd.map(lambda x: parseCSV(x)) 
```

```python
print("Number of entry in the rdd : %i" % listings_rdd.count())
```
```
Number of entry in the rdd : 64971
```

We are going to get the list of the columns of the RDD:

```python
n_features = len(listings_rdd.take(1)[0])
feature_list = list(zip(listings_rdd.take(1)[0], list(range(n_features))))
print("Total features: ", n_features)
pprint.pprint(feature_list)
```
```
Total features:  106
[('id', 0),
 ('listing_url', 1),
 ('scrape_id', 2),
 ('last_scraped', 3),
 ('name', 4),
 ('summary', 5),
 ('space', 6),
 ('description', 7),
 ('experiences_offered', 8),
 ('neighborhood_overview', 9),
 ('notes', 10),
 ('transit', 11),
 ('access', 12),
 ('interaction', 13),
 ('house_rules', 14),
 ('thumbnail_url', 15),
 ('medium_url', 16),
 ('picture_url', 17),
 ('xl_picture_url', 18),
 ('host_id', 19),
 ('host_url', 20),
 ('host_name', 21),
 ('host_since', 22),
 ('host_location', 23),
 ('host_about', 24),
 ('host_response_time', 25),
 ('host_response_rate', 26),
 ('host_acceptance_rate', 27),
 ('host_is_superhost', 28),
 ('host_thumbnail_url', 29),
 ('host_picture_url', 30),
 ('host_neighbourhood', 31),
 ('host_listings_count', 32),
 ('host_total_listings_count', 33),
 ('host_verifications', 34),
 ('host_has_profile_pic', 35),
 ('host_identity_verified', 36),
 ('street', 37),
 ('neighbourhood', 38),
 ('neighbourhood_cleansed', 39),
 ('neighbourhood_group_cleansed', 40),
 ('city', 41),
 ('state', 42),
 ('zipcode', 43),
 ('market', 44),
 ('smart_location', 45),
 ('country_code', 46),
 ('country', 47),
 ('latitude', 48),
 ('longitude', 49),
 ('is_location_exact', 50),
 ('property_type', 51),
 ('room_type', 52),
 ('accommodates', 53),
 ('bathrooms', 54),
 ('bedrooms', 55),
 ('beds', 56),
 ('bed_type', 57),
 ('amenities', 58),
 ('square_feet', 59),
 ('price', 60),
 ('weekly_price', 61),
 ('monthly_price', 62),
 ('security_deposit', 63),
 ('cleaning_fee', 64),
 ('guests_included', 65),
 ('extra_people', 66),
 ('minimum_nights', 67),
 ('maximum_nights', 68),
 ('minimum_minimum_nights', 69),
 ('maximum_minimum_nights', 70),
 ('minimum_maximum_nights', 71),
 ('maximum_maximum_nights', 72),
 ('minimum_nights_avg_ntm', 73),
 ('maximum_nights_avg_ntm', 74),
 ('calendar_updated', 75),
 ('has_availability', 76),
 ('availability_30', 77),
 ('availability_60', 78),
 ('availability_90', 79),
 ('availability_365', 80),
 ('calendar_last_scraped', 81),
 ('number_of_reviews', 82),
 ('number_of_reviews_ltm', 83),
 ('first_review', 84),
 ('last_review', 85),
 ('review_scores_rating', 86),
 ('review_scores_accuracy', 87),
 ('review_scores_cleanliness', 88),
 ('review_scores_checkin', 89),
 ('review_scores_communication', 90),
 ('review_scores_location', 91),
 ('review_scores_value', 92),
 ('requires_license', 93),
 ('license', 94),
 ('jurisdiction_names', 95),
 ('instant_bookable', 96),
 ('is_business_travel_ready', 97),
 ('cancellation_policy', 98),
 ('require_guest_profile_picture', 99),
 ('require_guest_phone_verification', 100),
 ('calculated_host_listings_count', 101),
 ('calculated_host_listings_count_entire_homes', 102),
 ('calculated_host_listings_count_private_rooms', 103),
 ('calculated_host_listings_count_shared_rooms', 104),
 ('reviews_per_month', 105)]
```

106 columns is a lot! I am going to keep only some of them according to the following criteria:
1. Our scenario is to compute a price for a listing. In this context, data about ratings and comments



