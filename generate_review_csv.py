import json
import matplotlib.pyplot as plt 
import numpy as np 
from numpy import genfromtxt
import csv
from sklearn.datasets import make_blobs
import argparse

BUSINESS_FILE = './yelp_academic_dataset_business.json'
CHECKIN_FILE = './yelp_academic_dataset_checkin.json'
REVIEW_FILE = './yelp_academic_dataset_review.json'
TIP_FILE = './yelp_academic_dataset_tip.json'
USER_FILE = './yelp_academic_dataset_user.json'

# Generates a csv file containing all the restaurants in Austin
def GetBIDForAustin(business_data_set):
    num_restaurants = {}
    city = ""
    for restaurant in business_data_set:
        curr_city = restaurant.get('city')
        if num_restaurants.get(curr_city) is None:
            num_restaurants[curr_city]=1
        else:
            num_restaurants[curr_city]+=1
    
    count = 0
    for key in num_restaurants:
        if num_restaurants[key] >= count:
            city = key
            count = num_restaurants[key]
            
    restaurants = []
    for restaurant in business_data_set:
        if restaurant.get('city') == city:
            bid = restaurant.get('business_id')
            latitude = restaurant.get('latitude')
            longitude = restaurant.get('longitude')
            # categories = restaurant.get('categories')
            r = [bid, latitude, longitude]

            restaurants.append(r)
    
    with open('austin_restaurants.csv', 'w') as file:
        wr = csv.writer(file,delimiter=';')
        for restaurant in restaurants:
            wr.writerow([restaurant])

    return city, restaurants
    
def GetReviewCSV(austin_businesses):
    austin_reviews = []
    with open(REVIEW_FILE) as rfile:
        count = 0
        for line in rfile:
            count+=1
            print(count)
            data = json.loads(line)
            for bid in austin_businesses:
                if str(data['business_id']) == str(bid[0]): 
                   
                    business_id = data['business_id']
                    review_text = data['text']
                    review = [business_id, review_text]
                    austin_reviews.append(review)

    with open('austin_reviews.csv', 'w') as new_file:
        wr = csv.writer(new_file)
        for review in austin_reviews:
            wr.writerow(review, delimiter=';')

def ReadFile(filepath):
    data_set = []
    with open(filepath) as file:
        for line in file:
            data = json.loads(line)
            data_set.append(data)
    
    with open('reviews.csv', 'w') as new_file:
        wr = csv.writer(new_file)
        for review in data_set:
            row = []
            for key in review:
                row.append(review[key])
            wr.writerow(row)
    return data_set

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS565 Project 2')
    parser.add_argument('--task', help = 'task to perform')
    parser.add_argument('--busID', nargs='?', help = 'filepath to business ID csv file')
    # parser.add_argument('', help = 'k', type=int)
    # parser.add_argument('init', type = str, help = 'random, k-means++ or 1d')
    args = parser.parse_args()
    if args.task == 'find_most_popular_city':
        business_data_set = ReadFile(BUSINESS_FILE)
        city, restaurants = GetBIDForAustin(business_data_set)
        # print(city)
    elif args.task == 'gen_review_csv':
        # print(restaurants)
        review_data_set = ReadFile(REVIEW_FILE)
        # GetReviewCSV(review_data_set)
    elif args.task == 'austin_reviews':
        austin_restaurants = []
        with open('./austin_restaurants.csv') as afile:
            for line in afile:
                r = line.replace('[','').replace(']','').replace("'",'').split(',')
                austin_restaurants.append(r)

        GetReviewCSV(austin_restaurants)


    # GetReviewCSV(REVIEW_FILE, restaurants)
    
    pass
