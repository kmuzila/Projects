# Databricks notebook source
# MAGIC %md
# MAGIC #Phase 2 Notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ##Phase 2 Summary

# COMMAND ----------

# MAGIC %md
# MAGIC Summary:
# MAGIC 
# MAGIC Jacquie: This week I completed the first round of preprocessing on airline data, completed the timestamp creation and converted time to UTC based on timezone of airport. I also created graphs of the categorical data we've chosen and identified which columns from the airline dataset that need to be one-hot encoded and started encoding them. Additionally I started working on building the custom cross validation function for the logistic regression baseline.
# MAGIC 
# MAGIC Matt: This week I finalized our join notebook and set up our team with a compiled, joined dataset to begin creating and training models. In completing the join, I compiled all the preprocessed airline data from Jacquie and the weather data from Karl, and creating my own table of keys from the station data. Based on the join architecture, the datasets were joined in the following manner. In order to link the flights and weather data, I chose to link the two tables utilizing StationID (which is a combination of USAF and WAN codes). There is a unique StationID per weather station. In order to get the airline data to have a StationID, I utilized the "Neighbor_call" field to obtain an FAA airport code for each weather station. I generated my own station_key table that links StationID with the airport FAA code and then joined those two tables together with the flight data to give each flight data record a stationID. I then joined the combined Airline_Station table with the weather table by stationID and based on all weather data within 6 hours of the departured flight time to ensure proper data completeness for the team. I then wrote the final joined data to a delta lake for quick querying and processing.
# MAGIC 
# MAGIC Kasha: This week, I competed the feature selection and preprocessing so I can set up the logistic regression model. The logistic regression model works. The only thing we have to do is change the data split to replicate day forward-chaining, Jacquie will be coding the forward-chaining. I also started working on the GBT model but that is not completed yet. 
# MAGIC 
# MAGIC Karl: This week, I exploded compressed data columns to create unique columns for each signal then proceeded to normalize the missing data signals through the use of a mask to identify each columns unique missing data signifier. Wrote a custom join to augment the final dataset with new data to improve the model training.

# COMMAND ----------

from pyspark.sql.functions import col,isnan,when,count,lit, to_date,lpad,date_format,rpad,regexp_replace,concat,to_utc_timestamp,to_timestamp
from pyspark.sql.types import IntegerType,BooleanType,DateType,StringType,TimestampType
from pyspark.sql import DataFrameNaFunctions, SparkSession
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pytz import timezone

from pyspark.sql import SparkSession, DataFrameNaFunctions
from pyspark.sql.functions import substring,year,month,dayofmonth,split

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler, PCA, VectorSlicer, Imputer
from pyspark.ml.regression import LinearRegression,DecisionTreeRegressor,RandomForestRegressor,GBTRegressor
from pyspark.ml.classification import LogisticRegression,GBTClassifier
from pyspark.ml.evaluation import RegressionEvaluator,BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator, CrossValidatorModel
from pyspark.ml.stat import ChiSquareTest


# COMMAND ----------

blob_container = "tm30container" # The name of your container created in https://portal.azure.com
storage_account = "w261tm30" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w261tm30" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "tm30key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

# Inspect the Mount's Final Project folder 
display(dbutils.fs.ls(f"{blob_url}/"))

# COMMAND ----------

#Choose the Delta Lake Path:

DELTALAKE_DATA_PATH = f"{blob_url}/2022-03-19_data_chkpt_6m"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #Database Join Summary

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##Station Data Curation

# COMMAND ----------

#Read Stations File

df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")
display(df_stations)

# COMMAND ----------

#Create station key table

#Filter station and remove country code from the FAA code
filt_station = df_stations.filter(df_stations.station_id == df_stations.neighbor_id).filter(df_stations.neighbor_call.substr(1,1) == 'K')
key_station = filt_station.select('station_id', filt_station.neighbor_name.alias('airport_name'), filt_station.neighbor_call.substr(2,4).alias('FAA_Code'))
display(key_station)
key_station.count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Flights Data Curation

# COMMAND ----------

# Read data for all flights

# df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*")
# display(df_airlines)

# Load data for 6m of flights in 2015
df_airlines_6m = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_6m/*")
display(df_airlines_6m)

# COMMAND ----------

aptz = spark.table("aptz_csv")
display(aptz.select("*"))
print(aptz.count())

# COMMAND ----------

#US only timezone from <https://en.wikipedia.org/wiki/List_of_tz_database_time_zones>
ustz = spark.table("ustimezones_csv").select("TIMEZONE")
display(ustz)

# COMMAND ----------

airport_tz_join = aptz.join(ustz, aptz.TIMEZONE == ustz.TIMEZONE).drop(aptz.TIMEZONE)
display(airport_tz_join)
print(airport_tz_join.count())

# COMMAND ----------

#Working on 6 months of data

df2 = df_airlines_6m.select([(count(when(isnan(c) | col(c).isNull(), c))/count(lit(1))).alias(c)
                    for c in df_airlines_6m.columns]).collect()
#identify columns with more than 50% nulls
remove_columns = [c for c in df_airlines_6m.columns if df2[0].__getitem__(c) > .5]

#remove all columns that have 50% or more nulls and are related to target value
remove_unncessary_columns = ['DEP_DELAY', 'DEP_DELAY_GROUP', 'TAXI_OUT', 'WHEELS_OFF', 'FLIGHTS', 'OP_CARRIER', 'OP_CARRIER_AIRLINE_ID', 'CANCELLED', 'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY', 'TAXI_IN', 'ARR_DELAY_GROUP', 'ORIGIN_AIRPORT_SEQ_ID', 'DIVERTED', 'AIR_TIME', 'DISTANCE_GROUP', 'DISTANCE', 'DEST_AIRPORT_SEQ_ID',  'ORIGIN_CITY_MARKET_ID', 'FLIGHTS', 'ARR_DEL15', 'ARR_DEL_NEW','QUARTER','DIV_AIRPORT_LANDINGS', 'WHEELS_ON', 'DEP_TIME_BLK','ARR_TIME_BLK', 'ACTUAL_ELAPSED_TIME', 'YEAR', 'DAY_OF_MONTH', 'DEST_CITY_MARKET_ID', 'ORIGIN_AIRPORT_ID', 'ORIGIN_STATE_FIPS','ORIGIN_WAC', 'DEST_AIRPORT_ID', 'ORIGIN_AIRPORT_ID', 'ORIGIN_STATE_FIPS', 'DES_WAC', 'DEST_STATE_FIPS', 'DEST_STATE_NM','ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'ORIGIN_STATE_NM','DEST_CITY_NAME', 'DEST_STATE_ABR', 'DEST_STATE_NM','DEST_WAC', 'DEP_TIME']

final_df_airlines = df_airlines_6m.drop(*remove_columns).drop(*remove_unncessary_columns)

#Create time of day field
final_df_airlines = final_df_airlines.withColumn('TIME_OF_DAY', when(final_df_airlines.CRS_DEP_TIME.between(500,1159), 'Morning')\
    .when(final_df_airlines.CRS_DEP_TIME.between(1200,1659), 'Afternoon')\
    .when(final_df_airlines.CRS_DEP_TIME.between(1700,2259), 'Evening')\
    .otherwise('Night'))


#clean up data
#convert integers to strings

final_df_airlines = final_df_airlines.withColumn("OP_CARRIER_FL_NUM",col("OP_CARRIER_FL_NUM").cast(StringType())) 
final_df_airlines = final_df_airlines.withColumn("CRS_DEP_TIME",col("CRS_DEP_TIME").cast(StringType()))


final_df_airlines = final_df_airlines.withColumn('FL_DATE',to_date(final_df_airlines.FL_DATE, 'yyyy-MM-dd'))
final_df_airlines = final_df_airlines.withColumn("FL_DATE", date_format("FL_DATE", "yyyy-dd-MM"))

#Pad missing 0 and and convert into timestamp
final_df_airlines = final_df_airlines.withColumn('CRS_DEP_TIME', lpad(final_df_airlines.CRS_DEP_TIME,4, '0'))
final_df_airlines = final_df_airlines.withColumn('DATE_TIME', concat(col('FL_DATE'),lit(" "),col('CRS_DEP_TIME')))
final_df_airlines = final_df_airlines.withColumn("DATE_TIME", to_timestamp("DATE_TIME", "yyyy-dd-MM HHmm"))
final_df_airlines = final_df_airlines.withColumn('UNIQUE_ID', concat(col('OP_UNIQUE_CARRIER'),lit("-"),col('OP_CARRIER_FL_NUM'),lit("-"),col('DATE_TIME')))
final_df_airlines = final_df_airlines.dropDuplicates((['UNIQUE_ID']))

#convert integers to strings
final_df_airlines = final_df_airlines.withColumn("DAY_OF_WEEK", when(final_df_airlines.DAY_OF_WEEK == "1","MONDAY") \
      .when(final_df_airlines.DAY_OF_WEEK == "2","TUESDAY") \
      .when(final_df_airlines.DAY_OF_WEEK == "3","WEDNESDAY") \
      .when(final_df_airlines.DAY_OF_WEEK == "4","THURSDAY") \
      .when(final_df_airlines.DAY_OF_WEEK == "5","FRIDAY") \
      .when(final_df_airlines.DAY_OF_WEEK == "6","SATURDAY") \
      .when(final_df_airlines.DAY_OF_WEEK == "7","SUNDAY"))

#convert integers to strings
final_df_airlines = final_df_airlines.withColumn("MONTH", when(final_df_airlines.MONTH == "1","JAN") \
      .when(final_df_airlines.MONTH == "2","FEB") \
      .when(final_df_airlines.MONTH == "3","MAR") \
      .when(final_df_airlines.MONTH == "4","APR") \
      .when(final_df_airlines.MONTH == "5","MAY") \
      .when(final_df_airlines.MONTH == "6","JUNE") \
      .when(final_df_airlines.MONTH == "7","JULY") \
      .when(final_df_airlines.MONTH == "8","AUG") \
      .when(final_df_airlines.MONTH == "9","SEPT") \
      .when(final_df_airlines.MONTH == "10","OCT") \
      .when(final_df_airlines.MONTH == "11","NOV") \
      .when(final_df_airlines.MONTH == "12","DEC"))

#JOIN for timezone and convert to UTC
final_df_airlines = final_df_airlines.join(aptz, final_df_airlines.ORIGIN == aptz.AIRPORT, 'left').select("*")
final_df_airlines = final_df_airlines.select('*', to_utc_timestamp(final_df_airlines.DATE_TIME, final_df_airlines.TIMEZONE).alias('UTC_TIMESTAMP'))

final_drop = ['FL_DATE','OP_CARRIER_FL_NUM', 'CRS_DEP_TIME', 'AIRPORT']

#remove all rows that don't have a target value
final_df_airlines = final_df_airlines.drop(*final_drop).dropna()

display(final_df_airlines)

# COMMAND ----------

#Filter out any international flights for the final flights dataset
final_df_airlines_us = final_df_airlines.join(airport_tz_join, (final_df_airlines.ORIGIN == airport_tz_join.AIRPORT), 'inner').drop(airport_tz_join.AIRPORT)
final_df_airlines_us = final_df_airlines_us.join(airport_tz_join, (final_df_airlines_us.DEST == airport_tz_join.AIRPORT), 'inner').drop(airport_tz_join.AIRPORT)
display(final_df_airlines_us)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Weather Data Curation

# COMMAND ----------

# Read All Weather Data
df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*")
# display(df_weather)

# Load data for 6m of flights in 2015
df_weather_6m = df_weather.filter(col('DATE') < "2015-07-01T00:00:00.000")
display(df_weather_6m)

# COMMAND ----------

#When Karl is finished, place truncated weather data here
small_cols = [
'STATION',
'DATE',
'SOURCE',
'LATITUDE',
'LONGITUDE',
'ELEVATION',
'NAME',
'REPORT_TYPE',
'CALL_SIGN',
'WND',
'CIG',
'VIS',
'TMP',
'DEW',
'SLP',
'GA1', # SKY COVER
'AA1', # RAIN
'AJ1'  # SNOW
]

filtered_weather = df_weather_6m.select(*small_cols).filter(df_weather_6m.REPORT_TYPE.isin(['FM-15','FM-16']))
filtered_weather = filtered_weather.select([when(col(c)=="", None).otherwise(col(c)).alias(c) for c in filtered_weather.columns])
display(filtered_weather)

# COMMAND ----------

#                                   .withColumn('year',year(filtered_weather['DATE'])) \
#                                   .withColumn('month',month(filtered_weather['DATE'])) \
#                                   .withColumn('day',dayofmonth(filtered_weather['DATE'])) \
#                                   .withColumn('time', date_format(col('DATE'), 'HH:mm:ss')) \

filtered_weather = filtered_weather.withColumn("wnd_dir_angle", f.split(col('WND'), ',').getItem(0)) \
                                  .withColumn("wnd_dir_qual", f.split(col('WND'), ',').getItem(1)) \
                                  .withColumn("wnd_type", f.split(col('WND'), ',').getItem(2)) \
                                  .withColumn("wnd_spd_rate", f.split(col('WND'), ',').getItem(3).cast(IntegerType())) \
                                  .withColumn("wnd_spd_qual", f.split(col('WND'), ',').getItem(4)) \
                                  .withColumn("cig_ceil_ht", f.split(col('CIG'), ',').getItem(0).cast(IntegerType())) \
                                  .withColumn("cig_ceil_qual", f.split(col('CIG'), ',').getItem(1)) \
                                  .withColumn("cig_ceil_det", f.split(col('CIG'), ',').getItem(2)) \
                                  .withColumn("cig_cavok", f.split(col('CIG'), ',').getItem(3)) \
                                  .withColumn("vis_dist", f.split(col('VIS'), ',').getItem(0).cast(IntegerType())) \
                                  .withColumn("vis_dist_qual", f.split(col('VIS'), ',').getItem(1)) \
                                  .withColumn("vis_dist_var", f.split(col('VIS'), ',').getItem(2)) \
                                  .withColumn("vis_dist_qual_var", f.split(col('VIS'), ',').getItem(3)) \
                                  .withColumn("tmp_air", f.split(col('TMP'), ',').getItem(0).cast(IntegerType())) \
                                  .withColumn("tmp_air_qual", f.split(col('TMP'), ',').getItem(1)) \
                                  .withColumn("dew_pnt_tmp", f.split(col('DEW'), ',').getItem(0).cast(IntegerType())) \
                                  .withColumn("dew_pnt_qual", f.split(col('DEW'), ',').getItem(1)) \
                                  .withColumn("slp_prs", f.split(col('SLP'), ',').getItem(0).cast(IntegerType())) \
                                  .withColumn("slp_prs_qual", f.split(col('SLP'), ',').getItem(1)) \
                                  .withColumn("aa1_prd_quant_hr", f.split(col('AA1'), ',').getItem(0)) \
                                  .withColumn("aa1_dp", f.split(col('AA1'), ',').getItem(1).cast(IntegerType())) \
                                  .withColumn("aa1_cond", f.split(col('AA1'), ',').getItem(2)) \
                                  .withColumn("aa1_qual", f.split(col('AA1'), ',').getItem(3)) \
                                  .withColumn("aj1_dim", f.split(col('AJ1'), ',').getItem(0).cast(IntegerType())) \
                                  .withColumn("aj1_cond", f.split(col('AJ1'), ',').getItem(1)) \
                                  .withColumn("aj1_qual", f.split(col('AJ1'), ',').getItem(2)) \
                                  .withColumn("aj1_eq_wtr_dp", f.split(col('AJ1'), ',').getItem(3).cast(IntegerType())) \
                                  .withColumn("aj1_eq_wtr_cond", f.split(col('AJ1'), ',').getItem(4)) \
                                  .withColumn("aj1_eq_wtr_cond_qual", f.split(col('AJ1'), ',').getItem(5)) \
                                  .withColumn("ga1_cov", f.split(col('GA1'), ',').getItem(0)) \
                                  .withColumn("ga1_cov_qual", f.split(col('GA1'), ',').getItem(1)) \
                                  .withColumn("ga1_bs_ht", f.split(col('GA1'), ',').getItem(2).cast(IntegerType())) \
                                  .withColumn("ga1_bs_ht_qual", f.split(col('GA1'), ',').getItem(3)) \
                                  .withColumn("ga1_cld", f.split(col('GA1'), ',').getItem(4)) \
                                  .withColumn("ga1_cld_qual", f.split(col('GA1'), ',').getItem(5))

# COMMAND ----------

#Change bad values to nulls
missing_mask = {
    "wnd_dir_angle" : "999",
    "wnd_type" : "9",
    "wnd_spd_rate" : "9999",
    
    "cig_ceil_ht" : "99999",
    "cig_ceil_det" : "9",
    "cig_cavok" : "9",
    
    "vis_dist" : "999999",
    "vis_dist_var" : "9",
    
    "tmp_air" : "+9999", 
    
    "dew_pnt_tmp" : "+9999",
    
    "slp_prs" : "99999",
    
    "aa1_prd_quant_hr" : "99",
    "aa1_dp" : "9999",
    "aa1_cond" : "9",
    
    "aj1_dim" : "9999",
    "aj1_cond" : "9",
    "aj1_eq_wtr_dp" : "999999",
    "aj1_eq_wtr_cond" : "9",
    
    "ga1_cov" : "99",
    "ga1_bs_ht" : "+9999",
    "ga1_cld" : "99"
}

# COMMAND ----------

filtered_weather = filtered_weather.select([when(col(c)=="", None).otherwise(col(c)).alias(c) for c in filtered_weather.columns])
for k,v in missing_mask.items():
    filtered_weather = filtered_weather.replace(k, value=None, subset=[v])
display(filtered_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##Full Join
# MAGIC 
# MAGIC With the data tables cleaned and set up with the proper keys, the final joining can be performed.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Station Data Keys

# COMMAND ----------

station_org = key_station.select(*(col(x).alias('org_' + x) for x in key_station.columns))
station_des = key_station.select(*(col(x).alias('des_' + x) for x in key_station.columns))
display(station_des)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Final columns to be shown

# COMMAND ----------

final_columns = ['UNIQUE_ID',
                 'UTC_TIMESTAMP',
                 'DATE',
                 'TIME_OF_DAY',
                 'STATION',
                 'NAME',
                 'MONTH',
                 'DAY_OF_WEEK',
                 'OP_UNIQUE_CARRIER',
                 'TAIL_NUM',
                 'ORIGIN',
                 'DEST',
                 'DEP_DEL15',
                 'DEP_DELAY_NEW',
                 'ARR_DELAY_NEW',
                 'CRS_ELAPSED_TIME',
                 'SOURCE',
                 'LATITUDE',
                 'LONGITUDE',
                 'ELEVATION',
                 'REPORT_TYPE',
                 'CALL_SIGN',
                  'WND',
                  'CIG',
                  'VIS',
                  'TMP',
                  'DEW',
                  'SLP',
                  'GA1',
                  'AA1',
                  'AJ1',
                  'wnd_dir_angle',
                  'wnd_dir_qual',
                  'wnd_type',
                  'wnd_spd_rate',
                  'wnd_spd_qual',
                  'cig_ceil_ht',
                  'cig_ceil_qual',
                  'cig_ceil_det',
                  'cig_cavok',
                  'vis_dist',
                  'vis_dist_qual',
                  'vis_dist_var',
                  'vis_dist_qual_var',
                  'tmp_air',
                  'tmp_air_qual',
                  'dew_pnt_tmp',
                  'dew_pnt_qual',
                  'slp_prs',
                  'slp_prs_qual',
                  'aa1_prd_quant_hr',
                  'aa1_dp',
                  'aa1_cond',
                  'aa1_qual',
                  'aj1_dim',
                  'aj1_cond',
                  'aj1_qual',
                  'aj1_eq_wtr_dp',
                  'aj1_eq_wtr_cond',
                  'aj1_eq_wtr_cond_qual',
                  'ga1_cov',
                  'ga1_cov_qual',
                  'ga1_bs_ht',
                  'ga1_bs_ht_qual',
                  'ga1_cld',
                  'ga1_cld_qual']

# COMMAND ----------

#Full data set join for airports and flights (6 months)

join_full_org = final_df_airlines_us.join(station_org, final_df_airlines_us.ORIGIN == station_org.org_FAA_Code, 'inner')
join_full_airports = join_full_org.join(station_des, join_full_org.DEST == station_des.des_FAA_Code, 'inner')
display(join_full_airports)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Rationale for 6 hours
# MAGIC 
# MAGIC > *Strategic traffic flow managers must plan hours in advance to influence long-haul flights. If the time needed for pre-departure planning and filing of amended flight plans is added to the airborne time intervals, predictions of convective weather impacts on airspace capacity are needed 4-8 hours in advance to influence long-haul flights and 2-6 hours in advance to influence shorter flights.*
# MAGIC 
# MAGIC **- FAA:  <https://www.faa.gov/nextgen/programs/weather/faq/>**
# MAGIC 
# MAGIC 6 hours was chosen because that's the earliest the FAA will make a decision for pre-departure planning

# COMMAND ----------

#Join Full dataset join for flights and weather -- NOTE: Only for Origin airport at the moment!
#Weather includes only data for up to 6 hours before the flight departure time

join_full = join_full_airports.join(filtered_weather, (join_full_airports.org_station_id == filtered_weather.STATION) & \
                              ((join_full_airports.UTC_TIMESTAMP.cast("long") - filtered_weather.DATE.cast("long"))/3600 <= 6.0) & \
                              ((join_full_airports.UTC_TIMESTAMP.cast("long") - filtered_weather.DATE.cast("long"))/3600 > 0.0), 'inner').select(*final_columns).cache()

join_full = join_full.withColumnRenamed(existing = 'UTC_TIMESTAMP', new = 'FLIGHT_UTC_DATE')
join_full = join_full.withColumnRenamed(existing = 'DATE', new = 'WEATHER_UTC_DATE')
display(join_full)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##Delta Lake
# MAGIC 
# MAGIC Delta lakes were used because of the database resiliency it provides (with built-in version control) and quick query capabilties. We want to improve the rate at which we access the data to clean and train the data.

# COMMAND ----------

#To ensure checkpoints are saved by the date and not easily overwritten:

now = str(datetime.date.today())

DELTALAKE_DATA_PATH_TODAY = f"{blob_url}/{now}_data_chkpt_6m"
DELTALAKE_DATA_PATH_TODAY

# COMMAND ----------

# # Define the input and output formats and paths and the table name (partitioned by date).
write_format = 'delta'
partition_by = 'FLIGHT_UTC_DATE'
save_path = DELTALAKE_DATA_PATH_TODAY

# Remove table if it exists
# dbutils.fs.rm(DELTALAKE_DATA_PATH_TODAY, recurse=True)

# # Write the data to its target.
join_full.write \
  .partitionBy(partition_by) \
  .format(write_format) \
  .save(save_path)

# COMMAND ----------

#Read from Delta Lake

# Define the input and output formats and paths and the table name.
read_format = 'delta'
load_path = DELTALAKE_DATA_PATH
save_path = '/tmp/delta/join'
# table_name = 'flights.weather300m'

# Load the data from its source.
join_eda = spark \
  .read \
  .format(read_format) \
  .load(load_path)

# Create the table.
# spark.sql("CREATE TABLE " + table_name + " USING DELTA LOCATION '" + save_path + "'")

# Review data
display(join_eda)

# COMMAND ----------

#To further optimize the delta lake table, we utilize the OPTIMIZE function and ZORDER based on the airport origin for each flight

display(spark.sql("DROP TABLE  IF EXISTS join_eda"))
 
display(spark.sql("CREATE TABLE join_eda USING DELTA LOCATION '" + DELTALAKE_DATA_PATH + "'"))
                  
display(spark.sql("OPTIMIZE join_eda ZORDER BY (ORIGIN)"))

display(spark.sql("OPTIMIZE delta.DELTALAKE_DATA_PATH ZORDER BY (ORIGIN)"))


# COMMAND ----------

# MAGIC %md
# MAGIC #EDA

# COMMAND ----------

fields_plot = ['DEP_TIME_BLK','ARR_TIME_BLK','OP_UNIQUE_CARRIER', 'DAY_OF_WEEK', 'MONTH', 'TIME_OF_DAY']

p_final_df_airlines = final_df_airlines.toPandas()

fig = plt.figure(figsize=(15,12))
fig.subplots_adjust(hspace=1.5, wspace=0.5)
p=1
for fld in fields_plot:
  ax = fig.add_subplot(5, 2, p)
  grouped = p_final_df_airlines.groupby(fld)['DEP_DELAY_NEW'].mean()
  sns.barplot(x=fld,y='DEP_DELAY_NEW',data = p_final_df_airlines, ax=ax)
  plt.xticks(rotation=70)
  p+=1

# COMMAND ----------

# MAGIC %md
# MAGIC #One-Hot Encoding

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors


# COMMAND ----------

indexers = [StringIndexer(inputCol=column, outputCol=column+"_index", ).fit(final_df_airlines) for column in list(set(final_df_airlines.columns)-set(['ORIGIN', 'DEST', 'DEP_DEL15', 'ARR_DELAY_NEW', 'CRS_ELAPSED_TIME', 'SOURCE', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'REPORT_TYPE', 'CALL_SIGN', 'QUALITY_CONTROL', 'WND', 'CIG', 'VIS', 'TMP', 'DEW', 'SLP', 'GA1', 'GE1', 'GF1', 'MA1', 'REM', 'GD1'])) ]

onehotencoder_DAY_OF_WEEK_vector = OneHotEncoder(inputCol="DAY_OF_WEEK_index", outputCol="DAY_OF_WEEK_vec")
onehotencoder_MONTH_vector = OneHotEncoder(inputCol="MONTH_index", outputCol="MONTH_vec")
onehotencoder_TAIL_NUM_vector = OneHotEncoder(inputCol="TAIL_NUM_index", outputCol="TAIL_NUM_vec")
onehotencoder_TIME_OF_DAY_vector = OneHotEncoder(inputCol="TIME_OF_DAY_index", outputCol="TIME_OF_DAY_vec")



# COMMAND ----------

pipeline = Pipeline(stages=indexers+ [onehotencoder_DAY_OF_WEEK_vector, onehotencoder_TAIL_NUM_vector, onehotencoder_MONTH_vector, onehotencoder_TIME_OF_DAY_vector])
final_df_airlines = pipeline.fit(final_df_airlines).transform(final_df_airlines)


# COMMAND ----------

display(final_df_airlines)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #Feature Selection

# COMMAND ----------

index_cols = [
 'SOURCE',
 'LATITUDE',
 'LONGITUDE',
 'ELEVATION',
 'CALL_SIGN',
 'TAIL_NUM'
]

cat_cols = [
 'wnd_dir_angle',
 'wnd_type',
 'cig_ceil_det',
 'cig_cavok',
 'vis_dist_var',
 'aa1_cond',
 'ga1_cov',
 'ga1_cld',
 'MONTH',
 'DAY_OF_WEEK',
 'ARR_DELAY_NEW'
]

cont_cols = [
 'wnd_spd_rate',
 'cig_ceil_ht',
 'vis_dist',
 'tmp_air',
 'dew_pnt_tmp',
 'slp_prs',
 'aa1_prd_quant_hr',
 'aa1_dp',
 'ga1_bs_ht'
]

qual_cols = [
 'wnd_dir_qual',
 'wnd_spd_qual',
 'cig_ceil_qual',
 'vis_dist_qual',
 'vis_dist_qual_var',
 'tmp_air_qual',
 'dew_pnt_qual',
 'slp_prs_qual',
 'aa1_qual',
 'aj1_qual',
 'aj1_eq_wtr_cond_qual',
 'ga1_cov_qual',
 'ga1_bs_ht_qual',
 'ga1_cld_qual',
 'QUALITY_CONTROL'
]

pred_cols = 'DEP_DEL15'

# COMMAND ----------

train, val, test = three_month_df.randomSplit([0.8,0.1,0.1], seed = 2020)

# COMMAND ----------

# These are just counts. Don't have to run this 
train_count = train.count()
val_count = val.count()
test_count = test.count()
total_count = train_count + val_count + test_count
print('three_month_train records: {}\n three_month_val records: {}\n  three_month_test records: {}\n total records: {}'.format(train_count, val_count, test_count, total_count) ) # Check the number of records after data split 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##Preprocessing 

# COMMAND ----------

indexers = list(map(lambda c: StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid = 'keep'), index_cols+cat_cols))
encoders = list(map(lambda c: OneHotEncoder(inputCol=c + "_idx", outputCol=c+"_class"), index_cols+cat_cols))
imputers = [Imputer(inputCols=cont_cols, outputCols=cont_cols)]
features = list(map(lambda c: c+"_class", index_cols+cat_cols)) + cont_cols
assembler = [VectorAssembler(inputCols=features, outputCol="features"), StringIndexer(inputCol=pred_cols, outputCol="label")]
scaler = [StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)]
log_reg = LogisticRegression(featuresCol='scaledFeatures', elasticNetParam=0.5, maxIter=10)
param_grid = ParamGridBuilder().addGrid(log_reg.regParam, [0.1, 0.01]).build()
log_reg_pipeline = Pipeline(stages=indexers+encoders+imputers+assembler+scaler+[log_reg])
cross_val = CrossValidator(estimator=log_reg_pipeline, estimatorParamMaps=param_grid, evaluator=BinaryClassificationEvaluator(), numFolds=5)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##Logistic Regression

# COMMAND ----------

cross_val_mod = cross_val.fit(train)

# COMMAND ----------

best_log_mod = cross_val_mod.bestModel
log_reg_sum = best_log_mod.stages[len(best_log_mod.stages)-1].summary

# COMMAND ----------

display(log_reg_sum.roc)

# COMMAND ----------

f_measure = log_reg_sum.fMeasureByThreshold

# COMMAND ----------

f_measure_max = f_measure.groupBy().max('F-Measure').select('max(F-Measure)').head()['max(F-Measure)']
f_measure_pand = f_measure.toPandas()
best_f_measure_thresh = float(f_measure_pand[f_measure_pand['F-Measure'] == f_measure_max] ["threshold"])
log_reg.setThreshold(best_f_measure_thresh)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##Gradient Boosted Tree

# COMMAND ----------

tail_cnt = three_month_df.select('TAIL_NUM').distinct().count() + 1
features = list(map(lambda c: c+"_idx", index_cols+cat_cols)) + cont_cols
assembler = [VectorAssembler(inputCols=features, outputCol="features"), StringIndexer(inputCol=pred_cols, outputCol="label")]
gbt_class = GBTClassifier(featuresCol="features", labelCol="label", lossType = "logistic", maxBins = tail_cnt, maxIter=20, maxDepth=5)
gbt_pipeline = Pipeline(stages=indexers+imputers+assembler+[gbt_class])

# COMMAND ----------

gbt_model = gbt_pipeline.fit(train)
