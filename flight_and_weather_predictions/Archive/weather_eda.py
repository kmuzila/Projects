# Databricks notebook source
# MAGIC %md # Datasets
# MAGIC 
# MAGIC Flight delays create problems in scheduling for airlines and airports, leading to passenger inconvenience, and huge economic losses. As a result there is growing interest in predicting flight delays beforehand in order to optimize operations and improve customer satisfaction. In this project, you will be predicting flight delays using the datasets provided. For now, the problem to be tackled in this project is framed as follows:
# MAGIC 
# MAGIC * Predict departure delay/no delay, where a delay is defined as 15-minute delay (or greater) with respect to the planned time of departure. This prediction should be done two hours ahead of departure (thereby giving airlines and airports time to regroup and passengers a heads up on a delay). 
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC - As you can imagine this problem could be  framed in many different ways leading to different products, engineering challenges, and metrics for success. In your project proposals please feel free to list out a couple of these alternatives and their potential benefits and challenges.
# MAGIC 
# MAGIC - The data for this project comes in the form of two (BIG) tables - these tables are in databricks. The starter notebook has everything you need to access the data. All data is on Databricks.
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC * [Final Project Assignment](https://docs.google.com/document/d/1JlLV3X9oSTmmZDJYsHgZDZ5pwqZeEgEcFDIJhZuW6BU/edit?usp=sharing)
# MAGIC * [Team 30 Project Notes](https://docs.google.com/document/d/1TeLYI8VF_9KC6a3AjVsx-DYgE2yS1bM-t5RKZPEmiLY/edit?usp=sharing)

# COMMAND ----------

# MAGIC %md ### Airlines and Weather
# MAGIC * During the first half of the project you will perform EDA and implement and fine tune baseline models based on the three or six months of flight data. There will be a mid-project presentation day when you will have the opportunity to discuss your approaches, and ask questions. 
# MAGIC * Each team member should have a clear set of deliverables. You may want to create a team charter and project plan (gantt chart).
# MAGIC * The second half of the project  will focus on the entire flight history for 2015-2019.

# COMMAND ----------

from pyspark.sql.functions import col

# COMMAND ----------

# Inspect the Mount's Final Project folder 
display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project"))

# COMMAND ----------

df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")
display(df_stations)

# COMMAND ----------

# MAGIC %md ## Flights table
# MAGIC 
# MAGIC This is a subset of the passenger flight's on-time performance data taken from the TranStats data collection available from the U.S. Department of Transportation (DOT)
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC * 1 Quarter:  For the first phase of the project you will focus on flights departing from two major US airports (ORD (Chicago O’Hare) and ATL (Atlanta) for the first quarter of 2015 (that is about 160k flights) .
# MAGIC * 2 Quarters:  A second dataset is also provided which is made up flights from the same two airports for the first six months of 2015.
# MAGIC * You can use either the three month or six month datasets for the initial phase of the project. 
# MAGIC * For the final phase of the project you will focus on the entire flight data departing from all major US airports for the 2015-2019 timeframe
# MAGIC 
# MAGIC ----
# MAGIC Overall the stats for this project dataset are:
# MAGIC * flight dataset from the US Department of Transportation containing flight information from 2015 to 2019
# MAGIC * (31,746,841 x 109 dataframe)
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC A Data Dictionary for this dataset is located here:
# MAGIC https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ 

# COMMAND ----------

# Load 2015 Q1 for Flights
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")
display(df_airlines)

# COMMAND ----------

# MAGIC %md ## Weather table 
# MAGIC 
# MAGIC As a frequent flier, we know that flight departures (and arrivals)  often get affected by weather conditions, so it makes sense to collect and process weather data corresponding to the origin and destination airports at the time of departure and arrival respectively and build features based upon this data. 
# MAGIC A weather table  has been pre-downloaded from the National Oceanic and Atmospheric Administration repository  to S3 in the form of  parquet files (thereby enabling pushdown querying and efficient joins). The weather data is for the period Jan 2015 – December 2019. 
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC Overall the stats for this project dataset are:
# MAGIC 
# MAGIC * weather dataset from the National Oceanic and Atmospheric Administration repository containing weather information from 2015 to 2019
# MAGIC * (630,904,436 x 177 dataframe)
# MAGIC * [Global Hourly](https://www.ncdc.noaa.gov/cdo-web/datasets#:~:text=Climate%20Station%20Summaries-,Global%20Hourly%20Data,-Global%20Summary%20of)
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC Data dictionary ([subset source](https://docs.google.com/spreadsheets/d/1zhKPJ-6Bn79wgpZmtbBQOBGgNzooMHymiBukT8fe9_U/edit?usp=sharing)): 
# MAGIC 
# MAGIC | Field name           | Type    | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
# MAGIC |----------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# MAGIC | stn                  | STRING  | Cloud - GSOD NOAA                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
# MAGIC | wban                 | STRING  | WBAN number where applicable--this is the historical "Weather Bureau Air Force Navy" number - with WBAN being the acronym                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
# MAGIC | date                 | DATE    | Date of the weather observations                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
# MAGIC | year                 | STRING  | The year                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
# MAGIC | mo                   | STRING  | The month                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
# MAGIC | da                   | STRING  | The day                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
# MAGIC | temp                 | FLOAT   | Mean temperature for the day in degrees Fahrenheit to tenths. Missing = 9999.9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
# MAGIC | count_temp           | INTEGER | Number of observations used in calculating mean temperature                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
# MAGIC | dewp                 | FLOAT   | Mean dew point for the day in degreesm Fahrenheit to tenths. Missing = 9999.9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
# MAGIC | count_dewp           | INTEGER | Number of observations used in calculating mean dew point                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
# MAGIC | slp                  | FLOAT   | Mean sea level pressure for the day in millibars to tenths. Missing = 9999.9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
# MAGIC | count_slp            | INTEGER | Number of observations used in calculating mean sea level pressure                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
# MAGIC | stp                  | FLOAT   | Mean station pressure for the day in millibars to tenths. Missing = 9999.9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
# MAGIC | count_stp            | INTEGER | Number of observations used in calculating mean station pressure                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
# MAGIC | visib                | FLOAT   | Mean visibility for the day in miles to tenths. Missing = 999.9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
# MAGIC | count_visib          | INTEGER | Number of observations used in calculating mean visibility                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
# MAGIC | wdsp                 | STRING  | Mean wind speed for the day in knots to tenths. Missing = 999.9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
# MAGIC | count_wdsp           | STRING  | Number of observations used in calculating mean wind speed                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
# MAGIC | mxpsd                | STRING  | Maximum sustained wind speed reported for the day in knots to tenths. Missing = 999.9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
# MAGIC | gust                 | FLOAT   | Maximum wind gust reported for the day in knots to tenths. Missing = 999.9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
# MAGIC | max                  | FLOAT   | Maximum temperature reported during the day in Fahrenheit to tenths--time of max temp report varies by country and region, so this will sometimes not be the max for the calendar day. Missing = 9999.9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
# MAGIC | flag_max             | STRING  | Blank indicates max temp was taken from the explicit max temp report and not from the 'hourly' data. * indicates max temp was derived from the hourly data (i.e., highest hourly or synoptic-reported temperature)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
# MAGIC | min                  | FLOAT   | Minimum temperature reported during the day in Fahrenheit to tenths--time of min temp report varies by country and region, so this will sometimes not be the min for the calendar day. Missing = 9999.9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
# MAGIC | flag_min             | STRING  | Blank indicates min temp was taken from the explicit min temp report and not from the 'hourly' data. * indicates min temp was derived from the hourly data (i.e., lowest hourly or synoptic-reported temperature)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
# MAGIC | prcp                 | FLOAT   | Total precipitation (rain and/or melted snow) reported during the day in inches and hundredths; will usually not end with the midnight observation--i.e., may include latter part of previous day. .00 indicates no measurable precipitation (includes a trace). Missing = 99.99 Note: Many stations do not report '0' on days with no precipitation--therefore, '99.99' will often appear on these days. Also, for example, a station may only report a 6-hour amount for the period during which rain fell. See Flag field for source of data                                                                                                                                                                                                                                                                                                     |
# MAGIC | flag_prcp            | STRING  | A = 1 report of 6-hour precipitation amount B = Summation of 2 reports of 6-hour precipitation amount C = Summation of 3 reports of 6-hour precipitation amount D = Summation of 4 reports of 6-hour precipitation amount E = 1 report of 12-hour precipitation amount F = Summation of 2 reports of 12-hour precipitation amount G = 1 report of 24-hour precipitation amount H = Station reported '0' as the amount for the day (eg, from 6-hour reports), but also reported at least one occurrence of precipitation in hourly observations--this could indicate a trace occurred, but should be considered as incomplete data for the day. I = Station did not report any precip data for the day and did not report any occurrences of precipitation in its hourly observations--it's still possible that precip occurred but was not reported |
# MAGIC | sndp                 | FLOAT   | Snow depth in inches to tenths--last report for the day if reported more thanonce. Missing = 999.9 Note: Most stations do not report '0' ondays with no snow on the ground--therefore, '999.9' will often appear on these days                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
# MAGIC | fog                  | STRING  | Indicators (1 = yes, 0 = no/not reported) for the occurrence during the day                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
# MAGIC | rain_drizzle         | STRING  | Indicators (1 = yes, 0 = no/not reported) for the occurrence during the day                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
# MAGIC | snow_ice_pellets     | STRING  | Indicators (1 = yes, 0 = no/not reported) for the occurrence during the day                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
# MAGIC | hail                 | STRING  | Indicators (1 = yes, 0 = no/not reported) for the occurrence during the day                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
# MAGIC | thunder              | STRING  | Indicators (1 = yes, 0 = no/not reported) for the occurrence during the day                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
# MAGIC | tornado_funnel_cloud | STRING  | Indicators (1 = yes, 0 = no/not reported) for the occurrence during the day                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |

# COMMAND ----------

# Load the 2015 Q1 for Weather
df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(col('DATE') < "2015-04-01T00:00:00.000")
display(df_weather)

# COMMAND ----------

# MAGIC %md # Phase Deliverables and Requirements
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC ## Phases  Summaries
# MAGIC   * **A brief summary of your progress by addressing the requirements for each phase**
# MAGIC     * **These should be concise, one or two paragraphs**
# MAGIC     * **There is a character limit in the form of 1000 for these**
# MAGIC * Phase 1
# MAGIC   * Describe datasets
# MAGIC   * Joins
# MAGIC   * Task
# MAGIC   * Metrics
# MAGIC * Phase 2
# MAGIC   * EDA
# MAGIC   * Scalability
# MAGIC   * Efficiency
# MAGIC   * Distributed/parallel Training
# MAGIC   * Scoring Pipeline
# MAGIC * Phase 3 part 1
# MAGIC   * Baseline Pipeline
# MAGIC * Phase 3 part 2
# MAGIC   * In class Presentation
# MAGIC * Phase 4
# MAGIC   * Select optimal algorithm
# MAGIC   * Fine tune
# MAGIC   * Submit a final report (research style)
# MAGIC * Phase 5
# MAGIC   * Final Presentations

# COMMAND ----------

# MAGIC %md ## Question Formulation
# MAGIC 
# MAGIC You should refine the question formulation based on the general task description you’ve been given, ie, predicting flight delays. This should include some discussion of why this is an important task from a business perspective, who the stakeholders are, etc.. Some literature review will be helpful to figure out how this problem is being solved now, and the State Of The Art (SOTA) in this domain. Introduce the goal of your analysis. What questions will you seek to answer, why do people perform this kind of analysis on this kind of data? Preview what level of performance your model would need to achieve to be practically useful. Discuss evaluation metrics.

# COMMAND ----------

# MAGIC %md ## EDA & Discussion of Challenges
# MAGIC 
# MAGIC Determine a handful of relevant EDA tasks that will help you make decisions about how you implement the algorithm to be scalable. Discuss any challenges that you anticipate based on the EDA you perform

# COMMAND ----------

# MAGIC %md ## Feature Engineering
# MAGIC 
# MAGIC Apply relevant feature transformations, dimensionality reduction if needed, interaction terms, treatment of categorical variables, etc.. Justify your choices.

# COMMAND ----------

# MAGIC %md ## Algorithm Exploration
# MAGIC 
# MAGIC Apply 2 to 3 algorithms to the training set, and discuss expectations, trade-offs, and results. These will serve as your baselines - do not spend too much time fine tuning these. You will want to use this process to select a final algorithm which you will spend your efforts on fine tuning.

# COMMAND ----------

# MAGIC %md ## Algorithm Implementation
# MAGIC 
# MAGIC Create your own toy example that matches the dataset provided and use this toy example to explain the math behind the algorithm that you will perform. Apply your algorithm to the training dataset and evaluate your results on the test set. 

# COMMAND ----------

# MAGIC %md ## Conclusions
# MAGIC 
# MAGIC report results and learnings for both the ML as well as the scalability.

# COMMAND ----------

# MAGIC %md ## Application of Course Concepts
# MAGIC 
# MAGIC Pick 3-5 key course concepts and discuss how your work on this assignment illustrates an understanding of these concepts.

# COMMAND ----------

# MAGIC %md # Phase 1
# MAGIC   * Describe datasets
# MAGIC   * Joins
# MAGIC   * Task
# MAGIC   * Metrics
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC #### Brief Summary
# MAGIC 
# MAGIC The weather table (df_weather) was sourced from the National Oceanic and Atmosphere Administration (NOAA) with each record comprised of airport weather stations summaries that were reported hourly. The information reported contains temperature extremes, precipitation, visibility, wind speed, and other data points. The table provides data points that would be relevant to determining flying conditions and understand trends in weather patterns. The data has a large number of columns that appears to be extremely sparse or completely empty and would be beneficial to remove. The data also appears to have multiple records from the same source created hourly with extremely high overlap with the main difference being the report type. We can probably further dedupe the dataset by removing report types that are not applicable to our needs. 

# COMMAND ----------

# MAGIC %md ### Explain data
# MAGIC   * simple exploratory analysis of various fields
# MAGIC     * semantic meaning
# MAGIC     * intrinsic meaning of ranges
# MAGIC     * null values
# MAGIC     * categorical/numerical
# MAGIC     * mean/std.dev
# MAGIC     * normalize
# MAGIC     * scale inputs
# MAGIC   * Identify any:
# MAGIC     * missing data
# MAGIC     * corrupt data
# MAGIC     * outlier data

# COMMAND ----------

# Import file location??
# Why is it so slow?
# df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*")
df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(col('DATE') < "2015-04-01T00:00:00.000")

# COMMAND ----------

df_weather.printSchema()

# COMMAND ----------

# CULDROSE, UK reporting twice per hour (1:50,:2:00) with two different report types (FM-12,FM-15) are both needed?
# FM-12 = SYNOP Report of surface observation form a fixed land station
# FM-15 = METAR Aviation routine weather report
# Could probably remove FM-12 if there are no useful data unique to the report type
display(df_weather)

# COMMAND ----------

# Which to use?
# display(df_weather.describe())
# df_weather.summary().show()

# COMMAND ----------

print("Weather Table's report types: ")
df_weather.select("REPORT_TYPE").distinct().show()

# COMMAND ----------

print(f"Weather Table's total distinct station count: ")
df_weather.select("STATION").distinct().count()

# COMMAND ----------

print(f"Weather Table's total distinct station count: ")
df_weather.select("NAME").distinct().count()

# COMMAND ----------

print(f'Weather Table size with potential duplicates: ')
df_weather.count()

# COMMAND ----------

dedup_weather = df_weather.drop_duplicates()
print(f'Weather Table size after dropping duplicates: ')
dedup_weather.count()

# COMMAND ----------

# Already included in summary above??
print(f'Weather Table earliest report date: ')
df_weather.agg({"DATE": "min"}).show()

# COMMAND ----------

print(f'Weather Table latest report date: ')
df_weather.agg({"DATE": "max"}).show()

# COMMAND ----------

# count nulls
display(df_weather.isna().mean().round(4) * 100)
display(df_weather.count() / len(df_weather))

# COMMAND ----------

# Heatmap for remaining features

# COMMAND ----------

# MAGIC %md #### Define the outcome
# MAGIC   * evaluation metric
# MAGIC   * target precisely
# MAGIC   * mathematical formulas

# COMMAND ----------

# MAGIC %md #### Ingest and represent the CSV files
# MAGIC   * Efficiency
# MAGIC   * File formats

# COMMAND ----------

# MAGIC %md #### Join relevant datasets
# MAGIC   * Describe what tables to join
# MAGIC   * Describe the workflow
# MAGIC     * keys
# MAGIC     * type of join
# MAGIC   * Steps to deal with potential missing values

# COMMAND ----------

# MAGIC %md #### Checkpoint the data
# MAGIC   * avoid wasting time
# MAGIC   * avoid wasting resources

# COMMAND ----------

# MAGIC %md #### Split the data
# MAGIC   * 3 datasets
# MAGIC     * train
# MAGIC     * validation
# MAGIC     * test
# MAGIC   * no leaks
# MAGIC     * ex: normalize using the training statistics
# MAGIC     * ex: Cross-validation in Time Series
# MAGIC       * very different to regular cross-validation

# COMMAND ----------

# MAGIC %md # Future

# COMMAND ----------

# MAGIC %md ## Phase 2
# MAGIC   * EDA
# MAGIC   * Scalability
# MAGIC   * Efficiency
# MAGIC   * Distributed/parallel Training
# MAGIC   * Scoring Pipeline

# COMMAND ----------

# MAGIC %md ## Phase 3 part 1
# MAGIC   * Baseline Pipeline

# COMMAND ----------

# MAGIC %md ## Phase 3 part 2
# MAGIC   * In class Presentation

# COMMAND ----------

# MAGIC %md ## Phase 4
# MAGIC   * Select optimal algorithm
# MAGIC   * Fine tune
# MAGIC   * Submit a final report (research style)

# COMMAND ----------

# MAGIC %md ## Phase 5
# MAGIC   * Final Presentations
