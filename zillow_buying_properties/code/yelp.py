#!/usr/bin/python3
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

class Yelp:

    def __init__(self, api_key):
        self.raw_businesses = {}
        self.parse_businesses = []
        self.all_businesses = None
        self.zip_totals = {}
        self.request = requests.session()
        self.base_url = "https://api.yelp.com/v3/"
        self.gql_ext = "graphql"
        self.json_ext = "businesses/search"
        self.api_key = api_key
        self.header = {"Authorization": f"Bearer {self.api_key}",}
        self.gql_content_type = "application/graphql"
        self.json_content_type = "application/json"

    def get_gql_total(self, zip_code):
        try:
            total_query = f"""{{search(location: "{zip_code}"){{total}}}}"""
            self.header['Content-Type'] = self.gql_content_type
            response = requests.post(url=self.base_url + self.gql_ext, data=total_query, headers=self.header)
            response.raise_for_status()
            total = int(response.json()['data']['search']['total'])
            self.zip_totals[zip_code] = total
        except requests.exceptions.HTTPError as http_error:
            pass
        if response.json().get('error',None):
            error_code = response.json()['error']['code']
        elif response.json().get('errors',None):
            error_code = response.json()['errors'][0]['extensions']['code']
        else:
            error_code = None
        return response.status_code, response.reason, error_code
    
    def get_json_total(self, zip_code):
        try:
            params = {'location': zip_code}
            self.header['Content-Type'] = self.json_content_type
            response = requests.get(url=self.base_url + self.json_ext, params=params, headers=self.header)
            response.raise_for_status()
            total = int(response.json()['total'])
            self.zip_totals[zip_code] = total
        except requests.exceptions.HTTPError as http_error:
            pass
        if response.json().get('error',None):
            error_code = response.json()['error']['code']
        elif response.json().get('errors',None):
            error_code = response.json()['errors'][0]['extensions']['code']
        else:
            error_code = None
        return response.status_code, response.reason, error_code
    
    def get_zip_total(self, zip_code):
        print('Attempting GQL Total Query')
        status,reason,error_code = self.get_gql_total(zip_code)
        if status == 429 and error_code == 'TOO_MANY_REQUESTS_PER_SECOND':
            time.sleep(2)
            self.get_zip_total(zip_code)
        elif status == 500 or error_code == 'DAILY_POINTS_LIMIT_REACHED':
            print('Attempting JSON Total Query')
            status,reason,error_code = self.get_json_total(zip_code)
        elif status == 400:
            print("Error Status: " + str(status))
            print("Error Reason: " + reason)
            print("API Error Code: " + error_code)
        return status, error_code

    def get_qgl_businesses(self, code, offset, limit=50):
        try:
            business_query = f"""{{search(location: "{code}", offset: {offset}, limit: {limit}) {{business {{name id alias is_claimed is_closed price review_count rating location {{formatted_address postal_code city state}}}}}}}}"""
            self.header['Content-Type'] = self.gql_content_type
            response = requests.post(url=self.base_url + self.gql_ext, data=business_query, headers=self.header)
            response.raise_for_status()
            query_businesses = response.json()['data']['search']['business']
            if code in self.raw_businesses:
                self.raw_businesses[code].extend(query_businesses)
            else:
                self.raw_businesses[code] = query_businesses
        except requests.exceptions.HTTPError as http_error:
            pass
        if response.json().get('error',None):
            error_code = response.json()['error']['code']
        elif response.json().get('errors',None):
            error_code = response.json()['errors'][0]['extensions']['code']
        else:
            error_code = None
        return response.status_code, response.reason, error_code

    def get_json_businesses(self, code, offset, limit=50):
        try:
            params = {'location': code, 'offset': offset, 'limit': limit}
            self.header['Content-Type'] = self.json_content_type
            response = requests.get(url=self.base_url + self.json_ext, params=params, headers=self.header)
            response.raise_for_status()
            query_businesses = response.json()['businesses']
            if code in self.raw_businesses:
                self.raw_businesses[code].extend(query_businesses)
            else:
                self.raw_businesses[code] = query_businesses
        except requests.exceptions.HTTPError as http_error:
            pass
        if response.json().get('error',None):
            error_code = response.json()['error']['code']
        elif response.json().get('errors',None):
            error_code = response.json()['errors'][0]['extensions']['code']
        else:
            error_code = None
        return response.status_code, response.reason, error_code

    def business_parse(self, business):
        business.pop('image_url', None)
        business.pop('url', None)
        business.pop('coordinates', None)
        business.pop('transactions', None)
        business['location'].pop('address1', None)
        business['location'].pop('address2', None)
        business['location'].pop('address3', None)
        business['location'].pop('country', None)
        if not isinstance(business.get('is_claimed'),bool):
            business['is_claimed'] = None
        flat_parent_categories = ", ".join(set(z.get("alias") for x in business.get("categories", []) for z in x.get("parent_categories", [])))
        business['parent_categories'] = flat_parent_categories
        flat_categories = ", ".join(set(x.get("alias") for x in business.get("categories", [])))
        business['categories'] = flat_categories
        json_address = business['location'].pop('display_address', None)
        if not json_address:
            address_string = "\n".join(x for x in json_address)
            business['location']['formatted_address'] = address_string
        business_df = pd.json_normalize(business) 
        return business_df

    def get_business_data(self, code, total):
        for offset in range(0,total,50):
            print(f"{code} offset is {offset} out of {total}")
            limit_diff = total - offset
            print('Attempting GQL Business Data Query')
            if limit_diff < 50:
                status,reason,error_code = self.get_qgl_businesses(code, offset, limit_diff)
            else:
                status,reason,error_code = self.get_qgl_businesses(code, offset)
        if status == 429 and error_code == 'TOO_MANY_REQUESTS_PER_SECOND':
            time.sleep(2)
            self.get_business_data(code, total)
        elif status == 500 or error_code == 'DAILY_POINTS_LIMIT_REACHED':
            print('Attempting JSON Business Data Query')
            if limit_diff < 50:
                status,reason,error_code = self.get_json_businesses(code, offset, limit_diff)
            else:
                status,reason,error_code = self.get_json_businesses(code, offset)
        elif status == 400:
            print("Error Status: " + str(status))
            print("Error Reason: " + reason)
            print("API Error Code: " + error_code)
        return status, error_code


    def parse_business_data(self, businesses):
        norm_businesses = []
        for business in businesses:
            flat_categories = ", ".join(set(x.get("alias") for x in business.get("categories", [])))
            flat_parent_categories = ", ".join(set(z.get("alias") for x in business.get("categories", []) for z in x.get("parent_categories", [])))
            business['categories'] = flat_categories
            business['parent_categories'] = flat_parent_categories
            norm_businesses.append(pd.json_normalize(business))

        businesses_df = pd.concat(norm_businesses, ignore_index=True)
        self.parse_businesses.append(businesses_df)
        return businesses_df


    def start(self, zip_code_list):
        for code in zip_code_list:
            print(f'Starting to Query {code} total')
            status = 0
            while status != 200:
                status, error_code = self.get_zip_total(code)
                print(f'Query Status Code: {status}')
                if error_code:
                    print(f'Query Error Code: {error_code}')
                if status == 400:
                    break
            
        for code, total in self.zip_totals.items():
            print(f'Starting to Query {code} business details')
            status = 0
            while status != 200:
                status, error_code = self.get_business_data(code, total)
                print(f'Query Status Code: {status}')
                if error_code:
                    print(f'Query Error Code: {error_code}')
                if status == 400:
                    break
            self.parse_business_data(self.raw_businesses[code]) 

        self.all_businesses = pd.concat(self.parse_businesses)
        return self.all_businesses