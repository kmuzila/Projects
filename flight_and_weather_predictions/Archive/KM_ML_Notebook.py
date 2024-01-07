# Databricks notebook source
# MAGIC %md
# MAGIC ##Import Packages

# COMMAND ----------

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

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler, PCA, VectorSlicer, Imputer
from pyspark.ml.regression import LinearRegression,DecisionTreeRegressor,RandomForestRegressor,GBTRegressor
from pyspark.ml.classification import LogisticRegression,GBTClassifier
from pyspark.ml.evaluation import RegressionEvaluator,BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator, CrossValidatorModel
from pyspark.ml.stat import ChiSquareTest,Correlation
from pyspark.ml.linalg import DenseMatrix, Vectors

# COMMAND ----------

# import mlflow
# print(mlflow.__version__)

# spark.conf.set("spark.databricks.mlflow.trackMLlib.enabled", "true")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##Cloud Storage initialization

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

# display(dbutils.fs.ls(f"{blob_url}/"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##Final Joined Augment

# COMMAND ----------

# Potential addition:
# 'AT1', PRESENT-WEATHER-OBSERVATION
# 'AU1', DAILY-PRESENT-WEATHER-OBSERVATION
# 'AW1', PRESENT-WEATHER-OBSERVATION
# 'AX1', PAST-WEATHER-OBSERVATION 
# Hail doesnt appear to have a code

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

# COMMAND ----------

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

df_weather = spark.read.format("delta").load(DELTALAKE_WEATHER_PATH)
test_df = spark.read.format("delta").load(DELTALAKE_DATA_PATH)

# COMMAND ----------

rpt_weather = df_weather.select(*small_cols).filter(df_weather.REPORT_TYPE.isin(['FM-15','FM-16']))
rpt_weather = rpt_weather.select([when(col(c)=="", None).otherwise(col(c)).alias(c) for c in rpt_weather.columns])

# COMMAND ----------

split_weather = rpt_weather.withColumn('year',year(rpt_weather['DATE'])) \
                                .withColumn('month',month(rpt_weather['DATE'])) \
                                .withColumn('day',dayofmonth(rpt_weather['DATE'])) \
                                .withColumn('time', date_format(col('DATE'), 'HH:mm:ss')) \
                                .withColumn("wnd_dir_angle", split(col('WND'), ',').getItem(0)) \
                                .withColumn("wnd_dir_qual", split(col('WND'), ',').getItem(1)) \
                                .withColumn("wnd_type", split(col('WND'), ',').getItem(2)) \
                                .withColumn("wnd_spd_rate", split(col('WND'), ',').getItem(3).cast(IntegerType())) \
                                .withColumn("wnd_spd_qual", split(col('WND'), ',').getItem(4)) \
                                .withColumn("cig_ceil_ht", split(col('CIG'), ',').getItem(0).cast(IntegerType())) \
                                .withColumn("cig_ceil_qual", split(col('CIG'), ',').getItem(1)) \
                                .withColumn("cig_ceil_det", split(col('CIG'), ',').getItem(2)) \
                                .withColumn("cig_cavok", split(col('CIG'), ',').getItem(3)) \
                                .withColumn("vis_dist", split(col('VIS'), ',').getItem(0).cast(IntegerType())) \
                                .withColumn("vis_dist_qual", split(col('VIS'), ',').getItem(1)) \
                                .withColumn("vis_dist_var", split(col('VIS'), ',').getItem(2)) \
                                .withColumn("vis_dist_qual_var", split(col('VIS'), ',').getItem(3)) \
                                .withColumn("tmp_air", split(col('TMP'), ',').getItem(0).cast(IntegerType())) \
                                .withColumn("tmp_air_qual", split(col('TMP'), ',').getItem(1)) \
                                .withColumn("dew_pnt_tmp", split(col('DEW'), ',').getItem(0).cast(IntegerType())) \
                                .withColumn("dew_pnt_qual", split(col('DEW'), ',').getItem(1)) \
                                .withColumn("slp_prs", split(col('SLP'), ',').getItem(0).cast(IntegerType())) \
                                .withColumn("slp_prs_qual", split(col('SLP'), ',').getItem(1)) \
                                .withColumn("aa1_prd_quant_hr", split(col('AA1'), ',').getItem(0).cast(IntegerType())) \
                                .withColumn("aa1_dp", split(col('AA1'), ',').getItem(1).cast(IntegerType())) \
                                .withColumn("aa1_cond", split(col('AA1'), ',').getItem(2)) \
                                .withColumn("aa1_qual", split(col('AA1'), ',').getItem(3)) \
                                .withColumn("aj1_dim", split(col('AJ1'), ',').getItem(0).cast(IntegerType())) \
                                .withColumn("aj1_cond", split(col('AJ1'), ',').getItem(1)) \
                                .withColumn("aj1_qual", split(col('AJ1'), ',').getItem(2)) \
                                .withColumn("aj1_eq_wtr_dp", split(col('AJ1'), ',').getItem(3).cast(IntegerType())) \
                                .withColumn("aj1_eq_wtr_cond", split(col('AJ1'), ',').getItem(4)) \
                                .withColumn("aj1_eq_wtr_cond_qual", split(col('AJ1'), ',').getItem(5)) \
                                .withColumn("ga1_cov", split(col('GA1'), ',').getItem(0)) \
                                .withColumn("ga1_cov_qual", split(col('GA1'), ',').getItem(1)) \
                                .withColumn("ga1_bs_ht", split(col('GA1'), ',').getItem(2).cast(IntegerType())) \
                                .withColumn("ga1_bs_ht_qual", split(col('GA1'), ',').getItem(3)) \
                                .withColumn("ga1_cld", split(col('GA1'), ',').getItem(4)) \
                                .withColumn("ga1_cld_qual", split(col('GA1'), ',').getItem(5))
split_weather = split_weather.select([when(col(c)=="", None).otherwise(col(c)).alias(c) for c in split_weather.columns])

# COMMAND ----------

for k,v in missing_mask.items():
    split_weather = split_weather.replace(k, value=None, subset=[v])

# COMMAND ----------

prob_cols = ('year','month','day','time')
split_weather = split_weather.drop(*prob_cols)
dup_cols = list(set(test_df.columns).intersection(set(split_weather.columns)))

# COMMAND ----------

split_join = test_df.join(split_weather, dup_cols).cache()
three_month_df = test_df.filter(col('FLIGHT_UTC_DATE').between('2019-10-01T00:00:00.000', '2019-12-31T00:00:00.000'))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Data Split

# COMMAND ----------

six_month_df = spark.read.format("delta").load(DELTALAKE_DATA_PATH)

# COMMAND ----------

# Temp fix for current dataset: 2022-03-19_data_chkpt_6m
six_month_df.withColumn("aa1_prd_quant_hr ",six_month_df.aa1_prd_quant_hr .cast(IntegerType()))

# COMMAND ----------

train, val, test = six_month_df.randomSplit([0.8,0.1,0.1], seed = 2020)

# COMMAND ----------

train_count = train.count()
val_count = val.count()
test_count = test.count()
total_count = train_count + val_count + test_count
print('three_month_train records: {}\n three_month_val records: {}\n  three_month_test records: {}\n total records: {}'.format(train_count, val_count, test_count, total_count) ) # Check the number of records after data split 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Feature Selection

# COMMAND ----------

#  'aj1_cond_idx', should have at least two distinct values
#  'aj1_eq_wtr_cond', should have at least two distinct values
#  'aj1_dim', should have at least two distinct values
#  'aj1_eq_wtr_dp', should have at least two distinct values

index_cols = [
 'SOURCE',
 'LATITUDE',
 'LONGITUDE',
 'ELEVATION',
 'CALL_SIGN',
 'TAIL_NUM',
 'OP_UNIQUE_CARRIER',
 'TIME_OF_DAY'
]

cat_cols = [
 'wnd_dir_angle',
 'wnd_type',
 'cig_ceil_det',
 'cig_cavok',
 'vis_dist_var',
 'aa1_cond', # RAIN
 'ga1_cov', # SKY COVER
 'ga1_cld', # SKY COVER
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
 'aa1_prd_quant_hr', # RAIN
 'aa1_dp', # RAIN
 'ga1_bs_ht' # SKY COVER
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
 'aa1_qual', # RAIN
 'aj1_qual', # SNOW
 'aj1_eq_wtr_cond_qual', # SNOW
 'ga1_cov_qual', # SKY COVER
 'ga1_bs_ht_qual', # SKY COVER
 'ga1_cld_qual', # SKY COVER
 'QUALITY_CONTROL'
]

pred_cols = 'DEP_DEL15'

# COMMAND ----------

categ_cols = [
 'TAIL_NUM',
 'OP_UNIQUE_CARRIER',
 'TIME_OF_DAY',
 'MONTH',
 'DAY_OF_WEEK',
 'ARR_DELAY_NEW',
 'wnd_type',
 'cig_cavok',
 'vis_dist_var'
]

numb_cols = [
 'wnd_dir_angle',
 'wnd_spd_rate',
 'cig_ceil_ht',
 'vis_dist',
 'tmp_air',
 'dew_pnt_tmp',
 'slp_prs'
]

feat_cols = categ_col + numb_col

# COMMAND ----------

six_month_df.select([(f.count(f.when(f.isnan(c) | f.col(c).isNull() | (f.col(c)==""), c))/six_month_df.count()).alias(c) for c, t in six_month_df.dtypes if t != "timestamp"]).toPandas()

# COMMAND ----------

heatmap_pand = six_month_df.select(*index_cols).toPandas()
heatmap_corr = heatmap_pand.corr(method='spearman')
sns.set(rc={'figure.figsize':(10,10)})
sns.heatmap(heatmap_corr, annot=True)
plt.show()

# COMMAND ----------

heatmap_pand = six_month_df.select(*cat_cols).toPandas()
heatmap_corr = heatmap_pand.corr(method='spearman')
sns.set(rc={'figure.figsize':(10,10)})
sns.heatmap(heatmap_corr, annot=True)
plt.show()

# COMMAND ----------

heatmap_pand = six_month_df.select(*cont_cols).toPandas()
heatmap_corr = heatmap_pand.corr(method='spearman')
sns.set(rc={'figure.figsize':(10,10)})
sns.heatmap(heatmap_corr, annot=True)
plt.show()

# COMMAND ----------

heatmap_pand = six_month_df.select(*categ_col).toPandas()
heatmap_corr = heatmap_pand.corr(method='spearman')
sns.set(rc={'figure.figsize':(10,10)})
sns.heatmap(heatmap_corr, annot=True)
plt.show()

# COMMAND ----------

heatmap_pand = six_month_df.select(*numb_cols).toPandas()
heatmap_corr = heatmap_pand.corr(method='spearman')
sns.set(rc={'figure.figsize':(10,10)})
sns.heatmap(heatmap_corr, annot=True)
plt.show()

# COMMAND ----------

heatmap_pand = six_month_df.select(*feat_cols).toPandas()
heatmap_corr = heatmap_pand.corr(method='spearman')
sns.set(rc={'figure.figsize':(10,10)})
sns.heatmap(heatmap_corr, annot=True)
plt.show()

# COMMAND ----------

fields_plot = ['DEP_TIME_BLK','ARR_TIME_BLK','OP_UNIQUE_CARRIER', 'DAY_OF_WEEK', 'MONTH', 'TIME_OF_DAY']

six_month_pand = six_month_df.toPandas()

fig = plt.figure(figsize=(20,15))
fig.subplots_adjust(hspace=1.5, wspace=0.5)
p=1
for fld in fields_plot:
    ax = fig.add_subplot(5, 2, p)
    grouped = six_month_pand.groupby(fld)['DEP_DEL15'].count().reset_index()
    delayed = six_month_pand[six_month_pand.DEP_DEL15 == 1].groupby(fld)['DEP_DEL15'].count().reset_index()
    delayed['DEP_DEL15'] = [i / j * 100 for i,j in zip(delayed['DEP_DEL15'], grouped['DEP_DEL15'])]
    grouped['DEP_DEL15'] = [i / j * 100 for i,j in zip(grouped['DEP_DEL15'], grouped['DEP_DEL15'])]
    bar1 = sns.barplot(x=fld,  y="DEP_DEL15", data=grouped, color='lightblue')
    bar2 = sns.barplot(x=fld, y="DEP_DEL15", data=delayed, color='darkblue')
    plt.xticks(rotation=70)
    p+=1
    top_bar = mpatches.Patch(color='lightblue', label='Not Delayed')
    bottom_bar = mpatches.Patch(color='darkblue', label='Delayed')
    plt.legend(handles=[top_bar, bottom_bar], loc='upper right', bbox_to_anchor=(1.25, 1.0))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##Algorithm Exploration

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Preprocessing 

# COMMAND ----------

indexers = list(map(lambda c: StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid = 'keep'), index_cols+cat_cols))
encoders = list(map(lambda c: OneHotEncoder(inputCol=c + "_idx", outputCol=c+"_class"), index_cols+cat_cols))
imputers = [Imputer(inputCols=cont_cols, outputCols=cont_cols)]
features = list(map(lambda c: c+"_class", index_cols+cat_cols)) + cont_cols
assembler = [VectorAssembler(inputCols=features, outputCol="features"), StringIndexer(inputCol=pred_cols, outputCol="label")]
scaler = [StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)]
pca = PCA(k=1000,inputCol='features',outputCol='features_pca')
log_reg = LogisticRegression(featuresCol='scaledFeatures', elasticNetParam=0.5, maxIter=10)
param_grid = ParamGridBuilder().addGrid(log_reg.regParam, [0.1, 0.01]).build()
log_reg_pipeline = Pipeline(stages=indexers+encoders+imputers+assembler+scaler+[log_reg])
cross_val = CrossValidator(estimator=log_reg_pipeline, estimatorParamMaps=param_grid, evaluator=BinaryClassificationEvaluator(), numFolds=5)

# COMMAND ----------

pca_model = pca.fit(train)

# COMMAND ----------

pca_train = pca_model.transform(train)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Logistic Regression

# COMMAND ----------

cross_val_mod = cross_val.fit(train)

# COMMAND ----------

len(best_log_mod.stages)

# COMMAND ----------

best_log_mod = cross_val_mod.bestModel
log_reg_sum = best_log_mod.stages[len(best_log_mod.stages)-1].summary

# COMMAND ----------

# vector_col = "corr_features"
# assembler = VectorAssembler(inputCols=df.columns, outputCol=vector_col)
# df_vector = assembler.transform(df).select(vector_col)

# # get correlation matrix
# matrix = Correlation.corr(df_vector, vector_col)

# def correlation_matrix(df, corr_columns, method='pearson'):
#     vector_col = "corr_features"
#     assembler = VectorAssembler(inputCols=corr_columns, outputCol=vector_col)
#     df_vector = assembler.transform(df).select(vector_col)
#     matrix = Correlation.corr(df_vector, vector_col, method)

#     result = matrix.collect()[0]["pearson({})".format(vector_col)].values
#     return pd.DataFrame(result.reshape(-1, len(corr_columns)), columns=corr_columns, index=corr_columns)

# def plot_corr_matrix(correlations,attr,fig_no):
#     fig=plt.figure(fig_no)
#     ax=fig.add_subplot(111)
#     ax.set_title("Correlation Matrix for Specified Attributes")
#     ax.set_xticklabels(['']+attr)
#     ax.set_yticklabels(['']+attr)
#     cax=ax.matshow(correlations,vmax=1,vmin=-1)
#     fig.colorbar(cax)
#     plt.show()

# plot_corr_matrix(corrmatrix, columns, 234)

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

f_measure_max

# COMMAND ----------

f_measure_pand

# COMMAND ----------

best_f_measure_thresh

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Gradient Boosted Tree

# COMMAND ----------

tail_cnt = three_month_df.select('TAIL_NUM').distinct().count() + 1
features = list(map(lambda c: c+"_idx", index_cols+cat_cols)) + cont_cols
assembler = [VectorAssembler(inputCols=features, outputCol="features"), StringIndexer(inputCol=pred_cols, outputCol="label")]
gbt_class = GBTClassifier(featuresCol="features", labelCol="label", lossType = "logistic", maxBins = tail_cnt, maxIter=20, maxDepth=5)
gbt_pipeline = Pipeline(stages=indexers+imputers+assembler+[gbt_class])

# COMMAND ----------

gbt_model = gbt_pipeline.fit(train)
