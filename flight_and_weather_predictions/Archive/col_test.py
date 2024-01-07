# Databricks notebook source
import datetime
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pytz import timezone

from pyspark.sql import SparkSession, DataFrameNaFunctions
from pyspark.sql.types import IntegerType,BooleanType,DateType,StringType,TimestampType
from pyspark.sql.functions import col, substring,isnan,when,count,lit, to_date,lpad,date_format,rpad,regexp_replace,concat,to_utc_timestamp,to_timestamp,year,month,dayofmonth,split
from pyspark.sql import functions as f

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler, PCA, VectorSlicer, Imputer
from pyspark.ml.regression import LinearRegression,DecisionTreeRegressor,RandomForestRegressor,GBTRegressor
from pyspark.ml.classification import LogisticRegression,GBTClassifier
from pyspark.ml.evaluation import RegressionEvaluator,BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator, CrossValidatorModel
from pyspark.ml.stat import ChiSquareTest,Correlation
from pyspark.ml.linalg import DenseMatrix, Vectors

# COMMAND ----------

blob_container = "tm30container" # The name of your container created in https://portal.azure.com
storage_account = "w261tm30" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w261tm30" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "tm30key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# now = str(datetime.date.today())
# Configure Delta Lake Path
DELTALAKE_WEATHER_PATH = f"{blob_url}/weather_data_1d"
DELTALAKE_DATA_PATH = f"{blob_url}/2022-03-19_data_chkpt_6m"

# COMMAND ----------

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
'AJ1', # SNOW
'KA1', # EXTREME TEMP
'AT1',
'AX1'
]

qual_sus = ['2','6']
qual_err = ['3','7']

# COMMAND ----------

df_weather = spark.read.format("delta").load(DELTALAKE_WEATHER_PATH)

# COMMAND ----------

col_weather = df_weather.select(*small_cols)

# COMMAND ----------

#Did not do this because I need the DATE column to join
#                                   .withColumn('year',year(filtered_weather['DATE'])) \
#                                   .withColumn('month',month(filtered_weather['DATE'])) \
#                                   .withColumn('day',dayofmonth(filtered_weather['DATE'])) \
#                                   .withColumn('time', date_format(col('DATE'), 'HH:mm:ss')) \

filtered_weather = col_weather.withColumn("wnd_dir_angle", when(f.split(col('WND'), ',').getItem(0) == "999", "").otherwise(f.split(col('WND'), ',').getItem(0))) \
                                  .withColumn("wnd_dir_qual", f.split(col('WND'), ',').getItem(1)) \
                                  .withColumn("wnd_type",  when(f.split(col('WND'), ',').getItem(2) == "9", "").otherwise(f.split(col('WND'), ',').getItem(2))) \
                                  .withColumn("wnd_spd_rate", when(f.split(col('WND'), ',').getItem(3) == "9999", 0).otherwise(f.split(col('WND'), ',').getItem(3).cast(IntegerType()))) \
                                  .withColumn("wnd_spd_qual", f.split(col('WND'), ',').getItem(4)) \
                                  .withColumn("wnd_ex", when(col("WND") == "", 0).otherwise(1)) \
                                  .withColumn("wnd_dir_is_qual", when(f.split(col('WND'), ',').getItem(1).isin(qual_err+qual_sus), 0).otherwise(1)) \
                                  .withColumn("wnd_spd_is_qual", when(f.split(col('WND'), ',').getItem(4).isin(qual_err+qual_sus), 0).otherwise(1)) \
                                  .withColumn("cig_ceil_ht", when(f.split(col('CIG'), ',').getItem(0) == "9999", 0).otherwise(f.split(col('CIG'), ',').getItem(0).cast(IntegerType()))) \
                                  .withColumn("cig_ceil_qual", f.split(col('CIG'), ',').getItem(1)) \
                                  .withColumn("cig_ceil_det", when(f.split(col('CIG'), ',').getItem(2) == "9", "").otherwise(f.split(col('CIG'), ',').getItem(2))) \
                                  .withColumn("cig_cavok", when(f.split(col('CIG'), ',').getItem(3) == "9", "").otherwise(f.split(col('CIG'), ',').getItem(3))) \
                                  .withColumn("cig_ex", when(col("CIG") == "", 0).otherwise(1)) \
                                  .withColumn("cig_cavok_bool", when(f.split(col('CIG'), ',').getItem(3) == "9", "").when(f.split(col('CIG'), ',').getItem(1) == 'N', 0).otherwise(1)) \
                                  .withColumn("cig_ceil_is_qual", when(f.split(col('CIG'), ',').getItem(1).isin(qual_err+qual_sus), 0).otherwise(1)) \
                                  .withColumn("vis_dist", when(f.split(col('VIS'), ',').getItem(0) == "999999", 0).otherwise(f.split(col('VIS'), ',').getItem(0).cast(IntegerType()))) \
                                  .withColumn("vis_dist_qual", f.split(col('VIS'), ',').getItem(1)) \
                                  .withColumn("vis_dist_var", when(f.split(col('VIS'), ',').getItem(2) == "9", "").otherwise(f.split(col('VIS'), ',').getItem(2))) \
                                  .withColumn("vis_dist_qual_var", f.split(col('VIS'), ',').getItem(3)) \
                                  .withColumn("vis_ex", when(col("VIS") == "", 0).otherwise(1)) \
                                  .withColumn("vis_dist_var_bool", when(f.split(col('VIS'), ',').getItem(1) == 'N', 0).otherwise(1)) \
                                  .withColumn("vis_dist_is_qual", when(f.split(col('VIS'), ',').getItem(1).isin(qual_err+qual_sus), 0).otherwise(1)) \
                                  .withColumn("vis_dist_is_qual_var", when(f.split(col('VIS'), ',').getItem(3).isin(qual_err+qual_sus), 0).otherwise(1)) \
                                  .withColumn("tmp_air", when(f.split(col('TMP'), ',').getItem(0) == "+9999",0).otherwise(f.split(col('TMP'), ',').getItem(0).cast(IntegerType()))) \
                                  .withColumn("tmp_air_qual", f.split(col('TMP'), ',').getItem(1)) \
                                  .withColumn("tmp_ex", when(col("TMP") == "", 0).otherwise(1)) \
                                  .withColumn("tmp_air_is_qual", when(f.split(col('TMP'), ',').getItem(1).isin(qual_err+qual_sus), 0).otherwise(1)) \
                                  .withColumn("dew_pnt_tmp", when(f.split(col('DEW'), ',').getItem(0) == "+9999",0).otherwise(f.split(col('DEW'), ',').getItem(0).cast(IntegerType()))) \
                                  .withColumn("dew_pnt_qual", f.split(col('DEW'), ',').getItem(1)) \
                                  .withColumn("dew_ex", when(col("DEW") == "", 0).otherwise(1)) \
                                  .withColumn("dew_pnt_is_qual", when(f.split(col('DEW'), ',').getItem(1).isin(qual_err+qual_sus), 0).otherwise(1)) \
                                  .withColumn("slp_prs", when(f.split(col('SLP'), ',').getItem(0) == "99999",0).otherwise(f.split(col('SLP'), ',').getItem(0).cast(IntegerType()))) \
                                  .withColumn("slp_prs_qual", f.split(col('SLP'), ',').getItem(1)) \
                                  .withColumn("slp_ex", when(col("SLP") == "", 0).otherwise(1)) \
                                  .withColumn("slp_prs_is_qual", when(f.split(col('SLP'), ',').getItem(1).isin(qual_err+qual_sus), 0).otherwise(1)) \
                                  .withColumn("aa1_prd_quant_hr", when(f.split(col('AA1'), ',').getItem(0) == "99",0).otherwise(f.split(col('AA1'), ',').getItem(0).cast(IntegerType()))) \
                                  .withColumn("aa1_dp", when(f.split(col('AA1'), ',').getItem(1) == "9999",0).otherwise(f.split(col('AA1'), ',').getItem(1).cast(IntegerType()))) \
                                  .withColumn("aa1_cond", when(f.split(col('AA1'), ',').getItem(2) == "9", "").otherwise(f.split(col('AA1'), ',').getItem(2))) \
                                  .withColumn("aa1_qual", f.split(col('AA1'), ',').getItem(3)) \
                                  .withColumn("aa1_ex", when(col("AA1") == "", 0).otherwise(1)) \
                                  .withColumn("aa1_is_qual", when(f.split(col('AA1'), ',').getItem(3).isin(qual_err+qual_sus), 0).otherwise(1)) \
                                  .withColumn("aj1_dim", when(f.split(col('AJ1'), ',').getItem(0) == "9999",0).otherwise(f.split(col('AJ1'), ',').getItem(0).cast(IntegerType()))) \
                                  .withColumn("aj1_cond", when(f.split(col('AJ1'), ',').getItem(1) == "9", "").otherwise(f.split(col('AJ1'), ',').getItem(1))) \
                                  .withColumn("aj1_qual", f.split(col('AJ1'), ',').getItem(2)) \
                                  .withColumn("aj1_eq_wtr_dp", when(f.split(col('AJ1'), ',').getItem(3) == "999999",0).otherwise(f.split(col('AJ1'), ',').getItem(3).cast(IntegerType()))) \
                                  .withColumn("aj1_eq_wtr_cond", when(f.split(col('AJ1'), ',').getItem(4) == "9", "").otherwise(f.split(col('AJ1'), ',').getItem(4))) \
                                  .withColumn("aj1_eq_wtr_cond_qual", f.split(col('AJ1'), ',').getItem(5)) \
                                  .withColumn("aj1_ex", when(col("AJ1") == "", 0).otherwise(1)) \
                                  .withColumn("aj1_is_qual", when(f.split(col('AJ1'), ',').getItem(2).isin(qual_err+qual_sus), 0).otherwise(1)) \
                                  .withColumn("aj1_eq_wtr_cond_is_qual", when(f.split(col('AJ1'), ',').getItem(5).isin(qual_err+qual_sus), 0).otherwise(1)) \
                                  .withColumn("ga1_cov", when(f.split(col('GA1'), ',').getItem(0) == "99", "").otherwise(f.split(col('GA1'), ',').getItem(0))) \
                                  .withColumn("ga1_cov_qual", f.split(col('GA1'), ',').getItem(1)) \
                                  .withColumn("ga1_bs_ht", when(f.split(col('GA1'), ',').getItem(2) == "+9999",0).otherwise(f.split(col('GA1'), ',').getItem(2).cast(IntegerType()))) \
                                  .withColumn("ga1_bs_ht_qual", f.split(col('GA1'), ',').getItem(3)) \
                                  .withColumn("ga1_cld", when(f.split(col('GA1'), ',').getItem(4) == "99", "").otherwise(f.split(col('GA1'), ',').getItem(4))) \
                                  .withColumn("ga1_cld_qual", f.split(col('GA1'), ',').getItem(5)) \
                                  .withColumn("ga1_ex", when(col("GA1") == "", 0).otherwise(1)) \
                                  .withColumn("ga1_cov_is_qual", when(f.split(col('GA1'), ',').getItem(1).isin(qual_err+qual_sus), 0).otherwise(1)) \
                                  .withColumn("ga1_bs_ht_is_qual", when(f.split(col('GA1'), ',').getItem(3).isin(qual_err+qual_sus), 0).otherwise(1)) \
                                  .withColumn("ga1_cld_qual", when(f.split(col('GA1'), ',').getItem(5).isin(qual_err+qual_sus), 0).otherwise(1)) \
                                  .withColumn("ka1_prd_quant", when(f.split(col('KA1'), ',').getItem(0) == "999",0).otherwise(f.split(col('KA1'), ',').getItem(0).cast(IntegerType()))) \
                                  .withColumn("ka1_code", when(f.split(col('KA1'), ',').getItem(1) == "9", "").otherwise(f.split(col('KA1'), ',').getItem(1))) \
                                  .withColumn("ka1_temp", when(f.split(col('KA1'), ',').getItem(2) == "+9999",0).otherwise(f.split(col('KA1'), ',').getItem(2).cast(IntegerType()))) \
                                  .withColumn("ka1_temp_qual", f.split(col('KA1'), ',').getItem(3)) \
                                  .withColumn("ka1_ex", when(col("KA1") == "", 0).otherwise(1)) \
                                  .withColumn("ka1_temp_is_qual", when(f.split(col('KA1'), ',').getItem(3).isin(qual_err+qual_sus), 0).otherwise(1)) \
                                  .withColumn("at1_src_elem", f.split(col('AT1'), ',').getItem(0)) \
                                  .withColumn("at1_wthr", f.split(col('AT1'), ',').getItem(1)) \
                                  .withColumn("at1_wthr_abrv", f.split(col('AT1'), ',').getItem(2)) \
                                  .withColumn("at1_qual", f.split(col('AT1'), ',').getItem(3)) \
                                  .withColumn("at1_ex", when(col("AT1") == "", 0).otherwise(1)) \
                                  .withColumn("at1_is_qual", when(f.split(col('AT1'), ',').getItem(3).isin(qual_err+qual_sus), 0).otherwise(1)) \
                                  .withColumn("ax1_atm", f.split(col('AX1'), ',').getItem(0)) \
                                  .withColumn("ax1_qual", f.split(col('AX1'), ',').getItem(1)) \
                                  .withColumn("ax1_prd_quant", when(f.split(col('AX1'), ',').getItem(2) == "99",0).otherwise(f.split(col('AX1'), ',').getItem(2).cast(IntegerType()))) \
                                  .withColumn("ax1_prd_qual", f.split(col('AX1'), ',').getItem(3)) \
                                  .withColumn("ax1_ex", when(col("AX1") == "", 0).otherwise(1)) \
                                  .withColumn("ax1_is_qual", when(f.split(col('AX1'), ',').getItem(1).isin(qual_err+qual_sus), 0).otherwise(1)) \
                                  .withColumn("ax1_prd_is_qual", when(f.split(col('AT1'), ',').getItem(3).isin(qual_err+qual_sus), 0).otherwise(1)) 

# COMMAND ----------

display(filtered_weather)

# COMMAND ----------

filtered_weather.columns

# COMMAND ----------


