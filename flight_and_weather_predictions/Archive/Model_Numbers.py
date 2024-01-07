# Databricks notebook source
from pyspark.sql.functions import col,isnan,when,count,lit, to_date,lpad,date_format,rpad,regexp_replace,concat,to_utc_timestamp,to_timestamp, countDistinct,unix_timestamp, row_number, when
from pyspark.sql.types import IntegerType,BooleanType,DateType,StringType,TimestampType
from pyspark.sql import DataFrameNaFunctions
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pytz import timezone
import datetime
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler, PCA, VectorSlicer, Imputer
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import percent_rank
from pyspark.sql import Window
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, LinearSVC, NaiveBayes,DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import functions as f

# COMMAND ----------

def data_pull(df, time_window = 'full', date_col='FLIGHT_UTC_DATE'):
    """Pull processed dataset"""
    if time_window == '2019':
        df = df.filter(f.year(col(date_col)) == 2019)
    elif time_window == '2018':
        df = df.filter(f.year(col(date_col)) == 2018)
    elif time_window == '2017':
        df = df.filter(f.year(col(date_col)) == 2017)
    elif time_window == '2016':
        df = df.filter(f.year(col(date_col)) == 2016) 
    
    #The commands below are for 2015 data
    elif time_window == '6m':
        df = df.filter(col(date_col) < "2015-07-01T00:00:00.000")  
    elif time_window == '3m':
        df = df.filter(col(date_col) < "2015-04-01T00:00:00.000")
        #comment this out if it takes too long
    
    print(f'{df.count():,} total records imported for the {time_window} dataset')
    return df

# COMMAND ----------

blob_container = "tm30container" # The name of your container created in https://portal.azure.com
storage_account = "w261tm30" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w261tm30" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "tm30key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

test_pq = spark.read.parquet(f"{blob_url}/2022-03-24_data_chkpt_PQ_full")

# COMMAND ----------

test_pq = test_pq.na.replace('', None, 'wnd_type').na.replace('', None, 'ga1_cld').na.replace('', 'NA', 'wnd_type').na.replace('', None, 'ga1_cov').withColumn('wnd_dir_angle',col('wnd_dir_angle').cast(IntegerType())).withColumn('ka1_temp', when(f.isnull('ka1_temp'), '0').when(f.col('ka1_temp') < 0, -1).otherwise('1'))

df_2015_2018 = test_pq.filter(col('FLIGHT_UTC_DATE') < "2019-01-01T00:00:00.000")

df_6m = data_pull(test_pq, time_window='6m', date_col='FLIGHT_UTC_DATE')

df_3m = data_pull(test_pq, time_window='3m', date_col='FLIGHT_UTC_DATE')

df_2019 = data_pull(test_pq, time_window='2019', date_col='FLIGHT_UTC_DATE')

# COMMAND ----------

#Chosen Model Columns

index_cols = ['UNIQUE_ID','FLIGHT_UTC_DATE', 'rank']

cat_cols = ['TIME_OF_DAY', 'MONTH', 'DAY_OF_WEEK', 'OP_UNIQUE_CARRIER', 'wnd_type', 'cig_ceil_is_qual', 'tmp_air_is_qual',  'slp_prs_is_qual', 'ga1_cov','ga1_cld', 'ga1_bs_ht_is_qual', 'wnd_spd_is_qual', 'ga1_cld_qual', 'dew_pnt_is_qual', 'ga1_cov_is_qual', 'aa1_is_qual', 'vis_dist_is_qual', 'TAIL_NUM']


cont_cols = ['ELEVATION', 'wnd_dir_angle', 'wnd_spd_rate', 'cig_ceil_ht', 'vis_dist', 'tmp_air', 'dew_pnt_tmp','slp_prs', 'aa1_prd_quant_hr', 'aa1_dp', 'ga1_bs_ht']

pred_cols = ['DEP_DEL15']



# COMMAND ----------

df_size = df_6m

# COMMAND ----------

#Preprocessing pipeline
train_test_window = df_size.withColumn("rank", percent_rank().over(Window.partitionBy().orderBy("FLIGHT_UTC_DATE")))
model_fields = index_cols + cont_cols + cat_cols + pred_cols
train_test_window = train_test_window.select(model_fields)
indexers = list(map(lambda c: StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid = 'keep'), cat_cols))
encoders = list(map(lambda c: OneHotEncoder(inputCol=c + "_idx", outputCol=c+"_class"), cat_cols))
imputers = [Imputer(inputCols=cont_cols, outputCols=cont_cols)]
preprocessing_pipeline = Pipeline(stages=indexers+encoders+imputers)
train_test_window = preprocessing_pipeline.fit(train_test_window).transform(train_test_window)

# COMMAND ----------

#Models FOR BASELINE ONLY
lr_class = LogisticRegression(featuresCol = 'features', labelCol = 'label')
rf_class = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
gbt_class = GBTClassifier(featuresCol="features", labelCol="label")
dt_class = DecisionTreeClassifier(featuresCol="features", labelCol="label")
lsvc_class = LinearSVC(featuresCol="features", labelCol="label")
nb_class = NaiveBayes(modelType='bernoulli',featuresCol="features", labelCol="label")
#gbt_class = GBTClassifier(featuresCol="scaledFeatures", labelCol="label", lossType = "logistic", maxBins = 32, maxIter=20, maxDepth=5)

# COMMAND ----------

def custom_CV(df, cont_cols, cat_cols, pipeline_model, kfolds):

  #Create Inner Preprocessing
    assembler_num = [VectorAssembler(inputCols=cont_cols, outputCol="scale_nums")]
    scaler = [StandardScaler(inputCol="scale_nums", outputCol="scaledFeatures", withStd=True, withMean=True)]

    features = list(map(lambda c: c+"_class", cat_cols))
    features.append('scaledFeatures')

    assembler = [VectorAssembler(inputCols=features, outputCol="features"), StringIndexer(inputCol='DEP_DEL15', outputCol="label")]
    #pca = PCA(k = 15, inputCol="features", outputCol="pcafeatures")
    
    scaled_pipeline = Pipeline(stages = assembler_num + scaler + assembler)

  #Create f1_score_list
    f1_score_list = []
    recall_list = []
    precision_list = []
  
  # Create Time Splits
    splits = 1/(kfolds + 1)
    cutoff = splits
    for split in range(kfolds):
        train_df = df.where(f"rank <= {cutoff}")
        test_df = df.where(f"rank > {cutoff} and rank <= {cutoff+splits}")
        cutoff += splits
        
        #Generate pipeline fit and tranform train and test based on pipeline
        train_pipeline = scaled_pipeline.fit(train_df)
        train_df_modified = train_pipeline.transform(train_df)
        test_df_modified = train_pipeline.transform(test_df)

        train_select_modified = train_df_modified.select('features','label')
        test_select_modified = test_df_modified.select('features','label')

        #Generate model 
        Model = pipeline_model.fit(train_df_modified)
        predict = Model.transform(test_df_modified)

        #calcuate f1 Score
        evaluatorf1 = MulticlassClassificationEvaluator(metricName='fMeasureByLabel', metricLabel=1, beta=.5)
        f1 = evaluatorf1.evaluate(predict)
        f1_score_list.append(f1)
        evaluator_recall = MulticlassClassificationEvaluator(metricName='recallByLabel', metricLabel=1)
        recall = evaluator_recall.evaluate(predict)
        recall_list.append(recall)
        evaluator_precision = MulticlassClassificationEvaluator(metricName='precisionByLabel', metricLabel=1)
        precision = evaluator_precision.evaluate(predict)
        precision_list.append(precision)
    return f1_score_list, recall_list, precision_list

# COMMAND ----------

f1_score, recall, precision = custom_CV(train_test_window, cont_cols, cat_cols, lr_class, 5)
print(f'Logistic Regression: {f1_score}, {recall}, {precision}')

# COMMAND ----------

print(np.mean([0.27133174354897993, 0.2750345098778125, 0.1583437984782448, 0.051240474783136564, 0.2754087906264455]), np.mean([0.08640328792476967, 0.12805500195481984, 0.042780161660512926, 0.010967499273392488, 0.0859296728857377]), np.mean([[0.5836014339553268, 0.38571346507352944, 0.4877133105802048, 0.6249391134924501, 0.6137409927942354]]))

# COMMAND ----------

f1_score, recall, precision = custom_CV(train_test_window, cont_cols, cat_cols, rf_class, 5)
print(f'Random Forest: {f1_score}, {recall}, {precision}')

# COMMAND ----------

print(f'Random Forest: {np.mean(f1_score)}, {np.mean(recall)}, {np.mean(precision)}')

# COMMAND ----------

f1_score, recall, precision = custom_CV(train_test_window, cont_cols, cat_cols, gbt_class, 5)
print(f'Gradient Boosted: {f1_score}, {recall}, {precision}')

# COMMAND ----------

f1_score, recall, precision = custom_CV(train_test_window, cont_cols, cat_cols, dt_class, 5)
print(f'Decision Tree: {f1_score}, {recall}, {precision}')

# COMMAND ----------

print(f'Decision Tree: {np.mean(f1_score)}, {np.mean(recall)}, {np.mean(precision)}')

# COMMAND ----------

f1_score, recall, precision = custom_CV(train_test_window, cont_cols, cat_cols, lsvc_class, 5)
print(f'Linear SVM: {f1_score}, {recall}, {precision}')

# COMMAND ----------

# f1_score, recall, precision = custom_CV(train_test_window, cont_cols, cat_cols, nb_class, 5)
# print(f'NaiveBayes: {f1_score}, {recall}, {precision}')

# COMMAND ----------

model_performance = {'Model': ['Logistic','Random Forest', 'Linear SVM'],'Cross Validation':['yes', 'yes', 'yes'], '2019 Tested':['no','no', 'no'], 'F1 Score Cross Validation':[.1192, .0836, .0004], 'F1 Score 2019 Test': [None, None, None]}
model_df = pd.DataFrame(model_performance, columns = model_performance.keys())
model_df

# COMMAND ----------



# COMMAND ----------


