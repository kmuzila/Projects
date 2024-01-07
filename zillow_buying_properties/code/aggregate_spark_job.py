#!/usr/bin/python3
import json
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, from_json, explode, split
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType, FloatType, IntegerType, ArrayType



"""
We want to create two tables, one table that stores the event of calling the API, 
and one event that appends the results of the API call. 
"""
def aggregate_request_event_schema():
    """
    root
     |-- Accept: string (nullable = true)
     |-- Content-Length: string (nullable = true)
     |-- Content-Type: string (nullable = true)
     |-- Host: string (nullable = true)
     |-- User-Agent: string (nullable = true)
     |-- zipcodes: string (nullable = true)
     |-- event_data: string (nullable = true)
     |-- event_type: string (nullable = true)
     |-- query_timestamp: string (nullable = true)
    """
    return StructType([
        StructField("Accept", StringType(), True),
        StructField("Content-Length", StringType(), True),
        StructField("Content-Type", StringType(), True),
        StructField("Host", StringType(), True),
        StructField("User-Agent", StringType(), True),
        StructField("zipcodes", StringType(), True),
        StructField("event_data", StringType(), True),
        StructField("event_type", StringType(), True),
        StructField("query_timestamp", StringType(), True)
    ])

@udf('boolean')
def is_zipcode_event(event_as_json):
    """
    udf for filtering events
    """
    event = json.loads(event_as_json)
    if event.get("event_type").startswith("get_"):
        return True
    return False

@udf('boolean')
def is_company_event(event_as_json):
    event = json.loads(event_as_json)
    event_type = event.get("event_type")
    return event_type == "get_yelp_data" or event_type == "get_zillow_data"

def main():
    """
    main
    
    """
    ##We open the spark session
    spark = SparkSession \
        .builder \
        .appName("ExtractEventsJob") \
        .enableHiveSupport() \
        .getOrCreate()    
    

    raw_events = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("subscribe", "events") \
    .option("multiline", "true") \
    .load()
    
    #raw_events.createOrReplaceTempView('raw_events_view')
    
    zipcode_data = raw_events \
        .filter(is_zipcode_event(raw_events.value.cast('string'))) \
        .select(raw_events.value.cast('string').alias('raw_event'),
                raw_events.timestamp.cast('string'),
                from_json(raw_events.value.cast('string'),
                          aggregate_request_event_schema()).alias('json')) \
        .select('raw_event', 'timestamp', 'json.*')    
        
#    yelp_zipcodes = zipcode_data.collect() \
#    .filter(zipcode_data.event_type == 'get_yelp_data') \
#    .select('zipcodes') \
#    .rdd.flatMap(lambda x: x)
#    
#    zillow_zipcodes = zipcode_data.collect() \
#    .filter(zipcode_data.event_type == 'get_zillow_data') \
#    .select('zipcodes') \
#    .rdd.flatMap(lambda x: x)
#    
#    zillow = spark \
#    .read \
#    .option('header', 'true') \
#    .csv('file:///w205/w205_project_3_karl_joe_kasha/zillow_2021_11_17.csv')
#     
#    zillow = zillow.filter(zillow.zipcode.isin(zillow_zipcodes))
#    sink2 = zillow.writeStream \
#        .format("parquet") \
#        .option("checkpointLocation", "/tmp/checkpoints_zillow_data") \
#        .option("path", "/tmp/zillow_data") \
#        .trigger(processingTime="60 seconds") \
#        .start()
#    
#    yelp = spark \
#    .read \
#    .option('header', 'true') \
#    .csv('file:///w205/w205_project_3_karl_joe_kasha/yelp_2021_11_17.csv')
#    
#    sink3 = yelp.filter(yelp["location.zipcode"].isin(yelp_zipcodes)) \
#        .writeStream \
#        .format("parquet") \
#        .option("checkpointLocation", "/tmp/checkpoints_yelp_data") \
#        .option("path", "/tmp/yelp_data") \
#        .trigger(processingTime="60 seconds") \
#        .start()
    
    sink = zipcode_data \
        .writeStream \
        .format("parquet") \
        .option("checkpointLocation", "/tmp/checkpoints_zipcode_data") \
        .option("path", "/tmp/zipcode_data") \
        .trigger(processingTime="60 seconds") \
        .start()
    
    sink.awaitTermination()
#   sink2.awaitTermination()
#   sink3.awaitTermination()
    

if __name__ == "__main__":
    main()