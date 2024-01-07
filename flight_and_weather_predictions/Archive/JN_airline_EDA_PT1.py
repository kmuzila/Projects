# Databricks notebook source
# Inspect the Mount's Final Project folder 
display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project"))

# COMMAND ----------

# Load 2015 Q1 for Flights
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")
display(df_airlines)

# COMMAND ----------

df2 = df_airlines.select([(count(when(isnan(c) | col(c).isNull(), c))/count(lit(1))).alias(c)
                    for c in df_airlines.columns]).collect()

# COMMAND ----------

# Load 2015 Q1 for Flights
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")
df2 = df_airlines.select([(count(when(isnan(c) | col(c).isNull(), c))/count(lit(1))).alias(c)
                    for c in df_airlines.columns]).collect()
#identify columns with more than 50% nulls
remove_columns = [c for c in df_airlines.columns if df2[0].__getitem__(c) > .5]

#remove all columns that have 50% or more nulls and are related to target value
remove_unncessary_columns = ['DEP_DELAY', 'DEP_DELAY_GROUP', 'TAXI_OUT', 'WHEELS_OFF', 'FLIGHTS', 'DEP_DELAY_NEW', 'OP_CARRIER', 'OP_CARRIER_AIRLINE_ID', 'CANCELLED', 'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY', 'TAXI_IN', 'ARR_DELAY_GROUP', 'ORIGIN_AIRPORT_SEQ_ID', 'DIVERTED', 'AIR_TIME', 'DISTANCE_GROUP', 'DISTANCE', 'DEST_AIRPORT_SEQ_ID',  'ORIGIN_CITY_MARKET_ID', 'FLIGHTS', 'ARR_DEL15', 'ARR_DEL_NEW','QUARTER','DIV_AIRPORT_LANDINGS', 'WHEELS_ON', 'DEP_TIME_BLK','ARR_TIME_BLK', 
'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'YEAR', 'DAY_OF_MONTH', 'DEST_CITY_MARKET_ID', 'ORIGIN_AIRPORT_ID', 'ORIGIN_STATE_FIPS','ORIGIN_WAC', 'DEST_AIRPORT_ID', 'ORIGIN_AIRPORT_ID', 'ORIGIN_STATE_FIPS', 'DES_WAC', 'DEST_STATE_FIPS', 'DEST_STATE_NM','ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'ORIGIN_STATE_NM','DEST_CITY_NAME', 'DEST_STATE_ABR', 'DEST_STATE_NM','DEST_WAC', 'DEP_TIME']

final_df_airlines = df_airlines.drop(*remove_columns).drop(*remove_unncessary_columns)

#clean up data
#convert string to date
final_df_airlines = final_df_airlines.withColumn('FL_DATE',to_date(final_df_airlines.FL_DATE, 'yyyy-MM-dd'))

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

final_df_airlines = final_df_airlines.withColumn("OP_CARRIER_FL_NUM",col("OP_CARRIER_FL_NUM").cast(StringType())) 
final_df_airlines = final_df_airlines.withColumn("CRS_DEP_TIME",col("CRS_DEP_TIME").cast(StringType()))

#Pad missing 0 and and convert into timestampe
final_df_airlines = final_df_airlines.withColumn('CRS_DEP_TIME', lpad(final_df_airlines.CRS_DEP_TIME,4, '0'))

#remove all rows that don't have a target value
final_df_airlines = final_df_airlines.dropna(subset='DEP_DEL15')

display(final_df_airlines)

# COMMAND ----------

len(remove_columns)

# COMMAND ----------

display(final_df_airlines)

# COMMAND ----------

#Create lists of numerical and categorical columns
quant_list = list()
cate_list = list()

def sep_columns(df):
  for column in df.dtypes:
    if column[1] == 'string':
      cate_list.append(column[0])
    else:
      quant_list.append(column[0])
  return quant_list, cate_list

q, c = sep_columns(final_df_airlines)

# COMMAND ----------

q

# COMMAND ----------

c

# COMMAND ----------

from pyspark.ml.stat import Correlation
from pyspark.ml.linalg import DenseMatrix, Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import *

assembler = VectorAssembler(inputCols=final_df_airlines.columns, 
outputCol="features",handleInvalid='keep')
df = assembler.transform(final_df_airlines).select("features")

# correlation will be in Dense Matrix
correlation = Correlation.corr(df,"features","pearson").collect()[0][0]

# To convert Dense Matrix into DataFrame
rows = correlation.toArray().tolist()
df = spark.createDataFrame(rows,final_df_airlines.columns)

# COMMAND ----------

airlinestest = final_df_airlines.toPandas()
dftest = airlinestest.corr(method='spearman')
sns.set(rc={'figure.figsize':(5,5)})
sns.heatmap(dftest, annot=True)
plt.show()

# COMMAND ----------

final_df_airlines.select('ARR_DELAY_NEW').summary().toPandas()

# COMMAND ----------

#identify what columns still have missing values
df3 = final_df_airlines.select([(count(when(isnan(c) | col(c).isNull(), c))/count(lit(1))).alias(c)
                    for c in final_df_airlines.columns]).toPandas()
df3
