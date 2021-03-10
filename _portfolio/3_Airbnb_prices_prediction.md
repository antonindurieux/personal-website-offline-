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

from urllib.request import urlopen
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
import json
import plotly.express as px
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

It is time to launch a [Spark context](https://spark.apache.org/docs/2.3.0/api/java/org/apache/spark/SparkContext.html){:target="_blank"} and a [Spark session](https://spark.apache.org/docs/2.2.1/api/python/pyspark.sql.html?highlight=sqlcontext#pyspark.sql.SparkSession):

```python
sc = SparkContext()
sparkSession = SparkSession(sc)
```

Now we can create an RDD from the data:

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
- Our scenario is to compute a price based on the features of the apartment, the flexibility of the booking rules and the ratings. I won't include the price by extra guest and the cleaning fees as I think they could themselves be dependent on the price.
- Some columns contain textual information which could be really valuable for our model but it could be complex to process and I won't get into NLP here yet. Thus I won't keep these columns.
- Some columns contain data that won't be useful (such as the host name and id, redundant location columns). There are also some columns moslty unfilled like "experiences_offered" and "square_feet".

We will see in the next section how to keep only the desired columns while building our DataFrame.

### 2.3 Creating the DataFrame

The DataFrame format will be easier to manipulate than the RDD. 

I start by cleaning the RDD. I remove the header and I ensure that all the rows that I keep are correctly made of 106 columns:

```python
# Removing the header
header = listings_rdd.take(1)
listings_rdd = listings_rdd.filter(lambda line: line != header[0])

# Filtering rows incorrectly formated
listings_rdd = listings_rdd.filter(lambda line: len(line)==106) 
```

So far all the data is represented as strings. Some functions will be required to cast the different fields while also handling missing values: 

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
        return float(field.strip('$').replace(',',''))

def toInt(field):
    """
    Cast the input to an integer
    """
    if field == '':
        return None
    else:
        return int(re.findall(r'-?\d+\.?\d*', field)[0])

def toString(field):
    """
    Cast the input to a string
    """
    if field == '':
        return None
    else:
        return str(field)
```

We can then apply these functions according to the data type of each column we want to keep:

```python
listings_cols = listings_rdd.map(lambda line: (
                                        toString(line[25]),
                                        toInt(line[26]),
                                        toString(line[28]),
                                        toString(line[36]),
                                        toString(line[39]),
                                        toFloat(line[48]),
                                        toFloat(line[49]),
                                        toString(line[51]),
                                        toString(line[52]), 
                                        toInt(line[53]), 
                                        toFloat(line[54]), 
                                        toInt(line[55]),
                                        toInt(line[56]), 
                                        toString(line[57]),
                                        toInt(line[65]),
                                        toInt(line[67]),
                                        toInt(line[68]),
                                        toInt(line[77]),
                                        toInt(line[78]),
                                        toInt(line[79]),
                                        toInt(line[80]),
                                        toInt(line[82]),
                                        toInt(line[83]),
                                        toInt(line[86]),
                                        toInt(line[87]),
                                        toInt(line[88]),
                                        toInt(line[89]),
                                        toInt(line[90]),
                                        toInt(line[91]),
                                        toInt(line[92]),
                                        toString(line[96]),
                                        toString(line[97]),
                                        toString(line[98]),
                                        toFloat(line[105]),
                                        toFloat(line[60])))
```

Now we have to define a schema for our DataFrame, where we specify the data type of each column:

```python
listingsSchema = StructType([StructField("host_response_time", StringType(), True), # 3rd argument: nullable
                             StructField("host_response_rate", IntegerType(), True),
                             StructField("host_is_superhost", StringType(), True),
                             StructField("host_identity_verified", StringType(), True),
                             StructField("neighbourhood", StringType(), True),
                             StructField("latitude", FloatType(), True),
                             StructField("longitude", FloatType(), True),
                             StructField("property_type", StringType(), True),
                             StructField("room_type", StringType(), True),
                             StructField("accomodates", IntegerType(), True),
                             StructField("bathrooms", FloatType(), True),
                             StructField("bedrooms", IntegerType(), True),
                             StructField("beds", IntegerType(), True),
                             StructField("bed_type", StringType(), True),
                             StructField("guests_included", IntegerType(), True),
                             StructField("minimum_night", IntegerType(), True),
                             StructField("maximum_night", IntegerType(), True),
                             StructField("availability_30", IntegerType(), True),
                             StructField("availability_60", IntegerType(), True),
                             StructField("availability_90", IntegerType(), True),
                             StructField("availability_365", IntegerType(), True),
                             StructField("number_of_reviews", IntegerType(), True),
                             StructField("number_of_reviews_ltm", IntegerType(), True),
                             StructField("review_scores_rating", IntegerType(), True),
                             StructField("review_scores_accuracy", IntegerType(), True),
                             StructField("review_scores_cleanliness", IntegerType(), True),
                             StructField("review_scores_checkin", IntegerType(), True),
                             StructField("review_scores_communication", IntegerType(), True),
                             StructField("review_scores_location", IntegerType(), True),
                             StructField("review_scores_value", IntegerType(), True),
                             StructField("instant_bookable", StringType(), True),
                             StructField("is_business_travel_ready", StringType(), True),
                             StructField("cancellation_policy", StringType(), True),
                             StructField("reviews_per_month", FloatType(), True),
                             StructField("price", FloatType(), True)]) 
```

We can finally build our DataFrame!

```python
listings_df = sparkSession.createDataFrame(listings_cols, listingsSchema)
```

```python
print("Number of rows : {}, number of columns : {}".format(listings_df.count(), len(listings_df.columns)))
```
```python
Number of rows : 64970, number of columns : 35
```

```python
# Check the first rows
listings_df.show(5)
```

|host_response_time|host_response_rate|host_is_superhost|host_identity_verified| neighbourhood|latitude|longitude|property_type|      room_type|accomodates|bathrooms|bedrooms|beds|     bed_type|guests_included|minimum_night|maximum_night|availability_30|availability_60|availability_90|availability_365|number_of_reviews|number_of_reviews_ltm|review_scores_rating|review_scores_accuracy|review_scores_cleanliness|review_scores_checkin|review_scores_communication|review_scores_location|review_scores_value|instant_bookable|is_business_travel_ready| cancellation_policy|reviews_per_month|price|
|------------------|------------------|-----------------|----------------------|--------------|--------|---------|-------------|---------------|-----------|---------|--------|----|-------------|---------------|-------------|-------------|---------------|---------------|---------------|----------------|-----------------|---------------------|--------------------|----------------------|-------------------------|---------------------|---------------------------|----------------------|-------------------|----------------|------------------------|--------------------|-----------------|-----|
|within a few hours|               100|                f|                     f|  Observatoire|48.83349|  2.31852|    Apartment|Entire home/apt|          2|      1.0|       0|   1|     Real Bed|              1|            2|           30|              1|              9|              9|             246|                8|                    1|                 100|                    10|                       10|                   10|                         10|                    10|                 10|               f|                       f|            flexible|             0.24| 60.0|
|    within an hour|               100|                f|                     t|Hôtel-de-Ville|  48.851|  2.35869|    Apartment|Entire home/apt|          2|      1.0|       0|   1|Pull-out Sofa|              1|            1|           90|              2|             17|             39|              70|              188|                   51|                  90|                     9|                        8|                    9|                          9|                    10|                  8|               t|                       f|strict_14_with_gr...|             1.51|115.0|
|    within an hour|               100|                f|                     t|Hôtel-de-Ville|48.85758|  2.35275|    Apartment|Entire home/apt|          4|      1.0|       2|   2|     Real Bed|              2|           10|           23|              0|             17|             36|             257|              252|                   28|                  94|                    10|                        9|                   10|                         10|                    10|                 10|               f|                       f|            moderate|             2.45|119.0|
|      within a day|               100|                f|                     t|         Opéra|48.87464|  2.34341|    Apartment|Entire home/apt|          2|      1.0|       1|   1|     Real Bed|              2|            6|          365|             17|             47|             77|             352|                6|                    0|                  96|                    10|                       10|                   10|                         10|                    10|                 10|               f|                       f|strict_14_with_gr...|             0.05|130.0|
|               N/A|              null|                f|                     f|  Ménilmontant|48.86528|  2.39326|    Apartment|Entire home/apt|          3|      1.0|       1|   1|     Real Bed|              1|            3|          365|              0|              0|              0|             255|                1|                    0|                 100|                  null|                     null|                 null|                       null|                  null|               null|               f|                       f|            moderate|             0.01| 90.0|

## 3. Data cleaning and exploration

In this part we will start by doing some basic data cleaning. Then we will explore some interesting features while handling the possible outliers.

### 3.1 Data cleaning
I start by checking some statistics on the numerical columns:

```python
numerical_features = ['latitude', 'longitude', 'accomodates', 'bathrooms', 'bedrooms', 'beds', 
                      'guests_included', 'minimum_night', 'maximum_night', 'number_of_reviews', 
                      'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
                      'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
                      'review_scores_value', 'reviews_per_month', 'price']
listings_df.describe(numerical_features).show()
```

|summary|           latitude|          longitude|       accomodates|         bathrooms|          bedrooms|              beds|   guests_included|    minimum_night|    maximum_night| number_of_reviews|review_scores_rating|review_scores_accuracy|review_scores_cleanliness|review_scores_checkin|review_scores_communication|review_scores_location|review_scores_value| reviews_per_month|             price|
|-------|-------------------|-------------------|------------------|------------------|------------------|------------------|------------------|-----------------|-----------------|------------------|--------------------|----------------------|-------------------------|---------------------|---------------------------|----------------------|-------------------|------------------|------------------|
|  count|              64970|              64970|             64970|             64913|             64890|             64527|             64970|            64970|            64970|             64970|               50584|                 50536|                    50547|                50515|                      50542|                 50516|              50514|             51685|             64970|
|   mean|  48.86417655246119| 2.3454444992811987| 3.053224565183931| 1.117819234976045| 1.086176606564956|1.6667906457761867|1.4869478220717254|5.277712790518701|860.1610281668462|19.430798830229335|   92.71753914281196|     9.575609466518918|        9.178052109917502|    9.659309116104128|          9.701594713307744|      9.65339694354264|  9.253296115928258|1.1937587307790405|118.25927351085116|
| stddev|0.01842583839748577|0.03353845181900523|1.5544321421288472|0.6426539066996059|0.9812627635224233|1.1372279652650317| 1.033266132663016|44.34684034149566|39236.91056061497| 39.77816733064706|    8.78379733208202|    0.8283955368649675|       1.1109781122859097|   0.7733217218546118|         0.7347838002163867|    0.6988392276829006| 0.9339444395657538|1.4250697191394348|173.76838840491894|
|    min|           48.81336|            2.22084|                 1|               0.0|                 0|                 0|                 1|                1|                1|                 0|                  20|                     2|                        2|                    2|                          2|                     2|                  2|              0.01|               0.0|
|    max|           48.90573|            2.47427|                17|              50.0|                50|                50|                16|             9999|         10000000|               828|                 100|                    10|                       10|                   10|                         10|                    10|                 10|             40.54|           10000.0|

I will remove the listings where there are no bathrooms, no bedrooms, no beds and a price equal to 0 as it seems inconsistent:

```python
listings_df = listings_df.filter((listings_df.bathrooms > 0) & 
                                (listings_df.bedrooms > 0) &
                                (listings_df.beds > 0) &
                                (listings_df.price > 0))
```

I will now check the different kind of property types:

```python
listings_df.groupBy(["property_type"]).count().show(n=50)
```
```
+--------------------+-----+
|       property_type|count|
+--------------------+-----+
|           Apartment|44986|
|           Townhouse|  230|
|         Guest suite|   31|
|Casa particular (...|    3|
|           Camper/RV|    1|
|      Boutique hotel|  850|
|                Loft| 1052|
|          Guesthouse|   84|
|              Hostel|   51|
|                Cave|    3|
|               Villa|   13|
|          Aparthotel|   21|
|               Other|   72|
|  Serviced apartment|  391|
|               Hotel|  158|
|             Cottage|    2|
|        Nature lodge|    1|
|               Igloo|    1|
|         Condominium| 1514|
|               House|  452|
|                Boat|   15|
|          Tiny house|   20|
|           Houseboat|   12|
|            Bungalow|    1|
|   Bed and breakfast|  208|
+--------------------+-----+
```

We can see that there are some fanciful property types (not sure if sleeping in an Igloo is feasible in Paris!). I will only keep the most represented property type to restrict the complexity of my model.

```python
# Filter on property type
property_type_list = ['Apartment', 'Townhouse', 'Boutique hotel', 'Loft', 
                      'Guesthouse', 'Hostel', 'Other', 'Serviced apartment', 'Hotel', 
                      'Condominium', 'House', 'Bed and breakfast']
listings_df = listings_df.filter(listings_df.property_type.isin(property_type_list))
```

Similarly, are there some kind of weird values for the room and bed types ?

```python
listings_df.groupBy(["room_type"]).count().show()
```
```
+---------------+-----+
|      room_type|count|
+---------------+-----+
|    Shared room|  392|
|     Hotel room| 1336|
|Entire home/apt|42129|
|   Private room| 6191|
+---------------+-----+
```

```python
listings_df.groupBy(["bed_type"]).count().show()
```
```
+-------------+-----+
|     bed_type|count|
+-------------+-----+
|       Airbed|   14|
|        Futon|  153|
|Pull-out Sofa|  832|
|        Couch|  181|
|     Real Bed|48868|
+-------------+-----+
```

Nothing seems too unrealistic so we won't filter the data based on these columns.

### 3.2 Prices distribution

We will check how the prices are ditributed:

```python
sns.set(rc={'figure.figsize': (9, 5)})

bins, counts = listings_df.select("price").rdd.flatMap(lambda x: x).histogram(100)

fig, ax = plt.subplots()
plt.hist(bins[:-1], bins=bins, weights=counts)
plt.title('Price distribution')
plt.xlabel('Price (€)')
plt.show()
```
![img1](/assets/images/airbnb_img1.png)

```python
listings_df.describe(['price']).show()
```
```
+-------+------------------+
|summary|             price|
+-------+------------------+
|  count|             50048|
|   mean|127.03037084398977|
| stddev|174.28140414102353|
|    min|               8.0|
|    max|           10000.0|
+-------+------------------+
```

First, notice in the figure that I made the assumption that the prices are in Euros (even if there was a $ sign next to each amount in the raw dataset). I will later show an evidence which strengthen this assumption.

We can see that the great majority of the prices are distributed toward the low values. I will treat the higher than 500€ prices as outliers and remove them from the data:

```python
listings_df = listings_df.filter(listings_df.price < 500)
```

```python
bins, counts = listings_df.select("price").rdd.flatMap(lambda x: x).histogram(100)

fig, ax = plt.subplots()
plt.hist(bins[:-1], bins=bins, weights=counts)
plt.title('Price distribution')
plt.xlabel('Price (€)')
plt.show()
```
![img2](/assets/images/airbnb_img2.png)

Now we can see that round prices (100, 150, 200...) are over-represented. It tends to strongly confirm that prices are in Euros: We wouldn't observe this pattern if prices had been converted from Euros to US Dollars.

### 3.3 Number of bedrooms

The number of bedrooms by listing should be an important explanatory variable of our model. We can study its distribution and check the average price by number of bedrooms.

```python
listings_df.groupBy(['bedrooms']).mean('price').sort(asc('bedrooms')).show()
```
```
+--------+------------------+
|bedrooms|        avg(price)|
+--------+------------------+
|       1| 89.04847614840989|
|       2| 151.3934151200587|
|       3|214.81846392552367|
|       4|  270.393536121673|
|       5|305.35897435897436|
|       6| 300.8181818181818|
|       7|             195.4|
|       9|             460.0|
|      38|             119.0|
|      50|              85.0|
+--------+------------------+
```
```python
# Removing listings with more than 10 bedrooms
listings_df = listings_df.filter((listings_df.bedrooms <= 10))

max_bedrooms = listings_df.agg({"bedrooms": "max"}).collect()[0][0]
bedrooms_histogram = listings_df.select(
    "bedrooms").rdd.flatMap(lambda x: x).histogram(max_bedrooms-1)

pd.DataFrame(list(zip(*bedrooms_histogram)),
             columns=['bin', 'total']).set_index('bin').plot(kind='bar')
plt.title("Number of bedrooms distribution")
plt.xlabel("Number of bedrooms")
plt.show()
```
![img3](/assets/images/airbnb_img3.png)

We can see that the average price is increasing from 1 bedroom to 5. There are very few listings with more than 4 bedrooms. I also removed listings with more than 10 bedrooms as it doesn't seem very realistic for Parisian apartments.

### 3.4 Ratings

We can check if the price seems related to the review scores:

```python
fig, axs = plt.subplots(1, 7, figsize=(25,3))

for i in range(7):
    axs[i].scatter(ratings_df[ratings_df.columns[i]], ratings_df.price, marker='.')
    axs[i].set_xlabel(ratings_df.columns[i])
axs[0].set_ylabel('price')
plt.show()
```
![img4](/assets/images/airbnb_img4.png)

### 3.5 Apartments locations

The listings locations should probably have a major impact. We can generate maps to see this effect.

```python
# Get a Pandas DataFrame from the latitude, longitude and price columns
coords_pandas_df = listings_df.select(['latitude', 'longitude', 'price']).toPandas()
```

```python
# Plot a scatter map of the prices
fig = px.scatter_mapbox(coords_pandas_df, 
                        lat="latitude", 
                        lon="longitude", 
                        color="price", 
                        color_continuous_scale="Agsunset", 
                        zoom=11,
                        mapbox_style="carto-positron", 
                        height=700, 
                        width=900)
fig.update_traces(marker=dict(size=4), selector=dict(mode='markers'))
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# Save the map as html
fig.write_html("paris_prices_scatter.html")
```

Here is the result below, the map is interactive so you can zoom in for a more detailed view:

<iframe width="1200" height="800" src="/assets/html/paris_prices_scatter.html" frameborder="0"></iframe>

As expected, we see that the price varies a lot according to the listings locations. Now we can do a map by neighbourhood to have an aggregate view:

```python
# Get a Pandas DataFrame of the neighbourhoods average prices
neighbourhood_price = listings_df.groupBy(['neighbourhood']).mean('price')
neighbourhood_price_df = neighbourhood_price.toPandas()
```

```python
# Get a geojson of the neighbourhoods
with urlopen("http://data.insideairbnb.com/france/ile-de-france/paris/2019-09-16/visualisations/neighbourhoods.geojson") as response:
    neighbourhoods_geojson = json.load(response)

# Assign ids in the geojson so it works with plotly
for i, elmt in enumerate(neighbourhoods_geojson['features']):
    neighbourhoods_geojson['features'][i]['id'] = neighbourhoods_geojson['features'][i]['properties']['neighbourhood']
```

```python
fig = px.choropleth_mapbox(neighbourhood_price_df, 
                           geojson=neighbourhoods_geojson, 
                           locations='neighbourhood', 
                           color='avg(price)',
                           color_continuous_scale="Agsunset",
                           mapbox_style="carto-positron", 
                           height=700, 
                           width=900,
                           zoom=11, 
                           center = {"lat": 48.8534, "lon": 2.3488},
                           opacity=0.5
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

fig.write_html("paris_prices_chloropleth.html")
```

<iframe width="1200" height="800" src="/assets/html/paris_prices_chloropleth.html" frameborder="0"></iframe>

We see that the average prices more than double according to the neighbourhood, so it should definitely be an important feature of the model.

