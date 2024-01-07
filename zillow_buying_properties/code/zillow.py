#!/usr/bin/python3
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

class Zillow:

    def __init__(self):
        self.properties = []
        self.request = requests.session()
        self.end = False
        self.base_url = "https://www.zillow.com/"
        self.ext = "ca/"
        self.page_num = 1
        self.params = "?searchQueryState=%7B%22usersSearchTerm%22%3A%22CA%22%2C%22mapBounds%22%3A%7B%22west%22%3A-127.755134328125%2C%22east%22%3A-110.858161671875%2C%22south%22%3A26.013271831159642%2C%22north%22%3A47.32089396066217%7D%2C%22mapZoom%22%3A6%2C%22regionSelection%22%3A%5B%7B%22regionId%22%3A9%2C%22regionType%22%3A2%7D%5D%2C%22isMapVisible%22%3Afalse%2C%22filterState%22%3A%7B%22ah%22%3A%7B%22value%22%3Atrue%7D%2C%22sort%22%3A%7B%22value%22%3A%22globalrelevanceex%22%7D%2C%22zo%22%3A%7B%22value%22%3Atrue%7D%7D%2C%22isListVisible%22%3Atrue%7D"
        self.header = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-encoding':'gzip, deflate, br',
    'accept-language': 'en-US,en;q=0.9',
    'upgrade-insecure-requests': '1',
    'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'
}

    def make_params(self):
        page = f"{self.page_num}_p/"
        # SearchQueryState:{
        search_query_state = "?searchQueryState=%7B"
        # "usersSearchTerm":"CA",
        user_search_term = "%22usersSearchTerm%22%3A%22CA%22%2C"
        # "mapBounds":{"west":-127.755134328125,"east":-110.858161671875,"south":26.013271831159642,"north":47.32089396066217},
        map_bounds = "%22mapBounds%22%3A%7B%22west%22%3A-127.755134328125%2C%22east%22%3A-110.858161671875%2C%22south%22%3A26.013271831159642%2C%22north%22%3A47.32089396066217%7D%2C"
        # "mapZoom":6,
        map_zoom = "%22mapZoom%22%3A6%2C"
        # "regionSelection":[{"regionId":9,"regionType":2}],
        region_selection = "%22regionSelection%22%3A%5B%7B%22regionId%22%3A9%2C%22regionType%22%3A2%7D%5D%2C"
        # "isMapVisible":false,
        is_map_visible = "%22isMapVisible%22%3Afalse%2C"
        # "filterState":{"ah":{"value":true},"sort":{"value":"globalrelevanceex"},"zo":{"value":true}},
        filter_state = "%22filterState%22%3A%7B%22ah%22%3A%7B%22value%22%3Atrue%7D%2C%22sort%22%3A%7B%22value%22%3A%22globalrelevanceex%22%7D%2C%22zo%22%3A%7B%22value%22%3Atrue%7D%7D%2C"
        # "isListVisible":true
        is_list_visible = "%22isListVisible%22%3Atrue"
        # ,"pagination":{"currentPage":2}
        pagination = f"%2C%22pagination%22%3A%7B%22currentPage%22%3A{self.page_num}%7D"
        # }
        closing_bracket = "%7D"

        if self.page_num != 1:
            self.params = page + search_query_state + user_search_term + map_bounds + map_zoom + region_selection + is_map_visible + filter_state + is_list_visible + pagination + closing_bracket

        return self.base_url + self.ext + self.params

    def get_house_data(self, house):
        zpid = house.get('id').split('_')[1]
        zillow_owned = house.find("div", class_="list-card-brokerage list-card-img-overlay").text.lower().split()[-2]
        price = house.find("div", class_="list-card-price").text.replace(',','')[1:]
        
        details = house.find("ul", class_="list-card-details").findAll('li')
        bedrooms = details[0].text.split()[0]
        bathrooms = details[1].text.split()[0]
        sqft = details[2].text.split()[0].replace(',','')

        full_address = house.find("address", class_="list-card-addr").text.lower()
        street_address = full_address.split(',')[0].strip()
        city = full_address.split(',')[1].strip()
        state = full_address.split(',')[-1].split()[0]
        zipcode = full_address.split(',')[-1].split()[1]

        house_dict = {'zpid':zpid, 'full_address':full_address, 'street_address':street_address, 'city':city, 'state':state, 'zipcode':zipcode, 'bedrooms':bedrooms, 'bathrooms':bathrooms, 'sqft':sqft, 'price':price, 'owner':zillow_owned}

        return house_dict

    def get_houses(self, url):
            response = requests.get(url, headers=self.header)
            soup = BeautifulSoup(response.content, 'html.parser')
            houses = soup.find_all("article", class_ = "list-card list-card-additional-attribution list-card_not-saved")
            last_page = int(soup.find("span", class_ = "Text-c11n-8-37-0__aiai24-0 eBcXID").text.split()[-1])

            for house in houses:
                parsed_house = self.get_house_data(house)
                self.properties.append(parsed_house)

            return last_page

    def scrape(self):
        while not self.end:
            full_url = self.make_params()
            last_page = self.get_houses(full_url)

            if self.page_num == last_page:
                self.end = True
            
            self.page_num += 1
            time.sleep(10)