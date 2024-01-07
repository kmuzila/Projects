#!/usr/bin/python3
import json
import datetime 
from kafka import KafkaProducer
from flask import Flask, request
import pandas as pd
import numpy as np
from yelp import Yelp
from zillow import Zillow


app = Flask(__name__)
producer = KafkaProducer(bootstrap_servers='kafka:29092')

# yelp_csv_name = '/w205/project-3-keirich/yelp_2021_11_17.csv'
yelp_csv_name = '/w205/w205_project_3_karl_joe_kasha/yelp_2021_11_17.csv'
yelp_df = pd.read_csv(yelp_csv_name)
yelp_df['price_count'] = yelp_df['price'].str.len()

zillow_df = pd.read_csv('/w205/w205_project_3_karl_joe_kasha/zillow_2021_11_17.csv')

zipcodes = zillow_df['zipcode'].unique()
zipcodes.sort()

def log_to_kafka(topic, event):
    event.update(request.headers)
    producer.send(topic, json.dumps(event).encode())

@app.route("/")
def default_response():
    default_event = {'event_type': 'default instructions'}
    log_to_kafka('default_call', default_event)
    return ('[[ADD INSTRUCTIONS HERE]]' )

@app.route("/zipcode")
def get_zipcodes():
    #Querying the scraped dataframe
    
    zipcodes_output = ' '.join(map(str, zipcodes))
    zipcode_agg = str(len(zipcodes))
    
    #Sending the event to kafka
    aggregate_request_event = {'event_type': 'get_zip_codes','zipcode' : zipcodes_output, 'zipcode_agg': zipcode_agg, 'query_timestamp' : datetime.datetime.now().strftime("%H:%M:%S.%f")}
    # print(zipcode_request_event)
    log_to_kafka('events', aggregate_request_event)

    #Final return to the user
    return zipcodes_output

@app.route("/<zipcode>")
def get_zipcode_agg(zipcode):
    #Querying the scraped dataframe
    
    yelp_agg = yelp_df[yelp_df['location.zip_code'] == 95835].groupby('location.zip_code').agg({'rating': ['count', 'mean', 'min', 'max', 'std', 'var'], 'review_count': ['sum', 'mean', 'min', 'max', 'std', 'var'], 'price_count' : ['count', 'mean', 'min', 'max', 'std', 'var']})
    zillow_agg = zillow_df[zillow_df['zipcode'] == 95835].groupby('zipcode').agg({'bedrooms': ['mean', 'min', 'max', 'std', 'var'], 'bathrooms': ['mean', 'min', 'max', 'std', 'var'], 'sqft' : ['mean', 'min', 'max', 'std', 'var'], 'price' : ['mean', 'min', 'max', 'std', 'var']})
    zipcode_agg = pd.concat([zillow_agg, yelp_agg], axis=1).reindex(zillow_agg.index)
    
    #Sending the event to kafka
    aggregate_request_event = {'event_type': 'get_zipcode_aggregate', 'zipcode': zipcode, 'event_data': zipcode_agg.to_string(), "query_timestamp" : datetime.datetime.now().strftime("%H:%M:%S.%f")}
    # print(zipcode_request_event)
    log_to_kafka('events', aggregate_request_event)

    #Final return to the user
    return zipcode_agg.to_string()

@app.route("/summary")
def get_summary_agg():
    #Querying the scraped dataframe
    
    zipcodes_output = ' '.join(map(str, zipcodes))
    
    yelp_agg = yelp_df.agg({'rating': ['describe'],'review_count': ['describe'],'price_count': ['describe']})
    zillow_agg = zillow_df.agg({'bedrooms': ['describe'],'bathrooms': ['describe'],'sqft': ['describe'],'price': ['describe']})
    zipcode_agg = pd.concat([zillow_agg, yelp_agg], axis=1).reindex(zillow_agg.index)
    
    #Sending the event to kafka
    aggregate_request_event = {'event_type': 'get_summary_agg', 'zipcode': zipcodes_output, 'event_data': zipcode_agg.to_string(), "query_timestamp" : datetime.datetime.now().strftime("%H:%M:%S.%f")}
    # print(zipcode_request_event)
    log_to_kafka('events', aggregate_request_event)

    #Final return to the user
    return zipcode_agg.to_string()

@app.route("/<zipcode>/refresh/<api_key>")
def get_refresh_data(zipcode, api_key):
    #Querying the scraped dataframe

    yelp = Yelp(api_key)
    
    if zipcode == "zipcode":
        yelp.start(zipcodes)
    else:
        yelp.start(np.array([np.int64(zipcode)]))
            
    today = datetime.date.today() 
    csv_name = f"/w205/w205_project_3_karl_joe_kasha/yelp_{today:%Y_%m_%d}.csv"
    yelp.all_businesses.to_csv(csv_name,index=False)
    yelp_csv_name = csv_name

        
    zipcodes_output = ' '.join(map(str, zipcodes))

    
    #Sending the event to kafka
    aggregate_request_event = {'event_type': 'get_refresh_data', 'zipcode': zipcode, 'event_data': csv_name, "query_timestamp" : datetime.datetime.now().strftime("%H:%M:%S.%f")}
    # print(zipcode_request_event)
    log_to_kafka('events', aggregate_request_event)

    #Final return to the user
    return csv_name

@app.route("/yelp/<zipcode>")
def get_yelp_data(zipcode = 0):
    zipcode = 92354
    yelp_json = yelp_df[yelp_df['location.zip_code'] == zipcode].to_json(orient='records')
    
    print(yelp_df[yelp_df['location.zip_code'] == zipcode].shape[0])
    #Sending the event to kafka
    aggregate_request_event = { \
        'event_type': 'get_yelp_data', \
        'zipcodes': zipcode, \
        'event_data': yelp_json, \
        "query_timestamp" : datetime.datetime.now().strftime("%H:%M:%S.%f") \
    }
    
    log_to_kafka('events', aggregate_request_event)
    return yelp_json

@app.route("/zillow/<zipcode>")
def get_zillow_data(zipcode):
    zipcode = 92557
    zillow_json = zillow_df[zillow_df['zipcode'] == zipcode].to_json(orient='records')
    
    #Sending the event to kafka
    aggregate_request_event = { \
        'event_type': 'get_zillow_data', \
        'zipcodes': zipcode, \
        'event_data': zillow_json, \
        "query_timestamp" : datetime.datetime.now().strftime("%H:%M:%S.%f") \
    }
    
    log_to_kafka('events', aggregate_request_event)
    return zillow_json

if __name__ == '__main__':
    app.debug=True
    app.run()