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

On this article I will show how to use [Spark](https://spark.apache.org/){:target="_blank"} to work on the Paris Airbnb listings. I will put myself in the shoes of someone who plan to put his parisian property on the famous lodging platform. A good question would thus be: given the features of my apartment (location, accomodation capacity, bedrooms...), as well as the flexibility of my booking rules (minimum and maximum number of nights, instant booking, cancellation rules...), what would be a price by night aligned with the competition?

This scenario will be a good excuse to:
- Show how to work with [PySpark](https://spark.apache.org/docs/latest/api/python/){:target="_blank"};
- Clean the dataset and explore some aspects of it;
- Produce a price model.

The iPython notebook of this project can be found [here](https://github.com/antonindurieux/Airbnb-price-analysis-with-Spark) (in french).

## 1. The data

The data will be downloaded from http://insideairbnb.com/. As specified on the website, "Inside Airbnb is an independent, non-commercial set of tools and data that allows you to explore how Airbnb is really being used in cities around the world". There is a lot of very interesting and valuable data on this website, but we must keep in mind that the data is not from "official sources" in case of inconsistencies.

I will use the Paris data, and more specifically the listings csv file which gives the characteristics of every listing in the city.

## 2. From the raw data to a Spark DataFrame

I will start by downloading and de-zipping the data. I will then show how to load it in a RDD then to a DataFrame. An RDD is a [Resilient Distributed Datasets](https://spark.apache.org/docs/3.1.1/rdd-programming-guide.html#resilient-distributed-datasets-rdds){:target="_blank"}, the fundamental data structure of Spark. A SPark DataFrame ["is conceptually equivalent to a table in a relational database or a data frame in R/Python, but with richer optimizations under the hood"](https://spark.apache.org/docs/latest/sql-programming-guide.html){:target="_blank"}. At the time of this writing, Python still did not support another Spark data structure, the [Datasets](https://spark.apache.org/docs/latest/sql-programming-guide.html){:target="_blank"}.





