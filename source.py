import json
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
from numpy import genfromtxt
import csv
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import neighbors
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split 
import argparse

BUSINESS_FILE = './yelp_academic_dataset_business.json'
CHECKIN_FILE = './yelp_academic_dataset_checkin.json'
REVIEW_FILE = './yelp_academic_dataset_review.json'
TIP_FILE = './yelp_academic_dataset_tip.json'
USER_FILE = './yelp_academic_dataset_user.json'
    
def GetReviewCSV(austin_businesses):
    austin_reviews = {}
    austin_ratings = {}
    with open(REVIEW_FILE) as rfile:
        count = 0
        for line in rfile:
            data = json.loads(line)
            business_id = data['business_id']

            if business_id in austin_businesses:
                review_text = data['text'].replace('\n', '').replace(';','.').strip()
                rating = data['stars']
                # print(review_text)
                if business_id in austin_reviews:
                    austin_reviews[business_id] += " "
                    austin_reviews[business_id] += review_text.replace('\n','')
                else:
                    austin_reviews[business_id] = review_text

                if business_id in austin_ratings:
                    austin_ratings[business_id].append(rating)
                else:
                    austin_ratings[business_id] = [rating]

    with open('austin_reviews.csv', 'w') as new_file:
        wr = csv.writer(new_file, delimiter=';')
        for business in austin_reviews:
            avg_rating = sum(austin_ratings[business]) / len(austin_ratings[business])
            print(avg_rating)
            wr.writerow([[business],[austin_reviews[business]],[avg_rating]])
    return austin_reviews

def GetBusinessIDCsv():
    austin_businesses = {}
    with open(BUSINESS_FILE) as file:
        for line in file:
            data = json.loads(line)
            if data.get('city') == 'Austin':
                bid = data.get('business_id')
                latitude = data.get('latitude')
                longitude = data.get('longitude')
                categories = data.get('categories')
                austin_businesses[bid] = [bid, latitude, longitude, categories]
    
    with open('austin_businesses.csv', 'w') as file:
        wr = csv.writer(file,delimiter=';')
        for b in austin_businesses:
            wr.writerow(austin_businesses[b])

    return austin_businesses

def k_means():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS565 Project 2')
    parser.add_argument('--task', help = 'task to perform')
    parser.add_argument('--busID', nargs='?', help = 'filepath to business ID csv file')
    parser.add_argument('--truek', nargs='?',help = 'k', type=int)
    # parser.add_argument('init', type = str, help = 'random, k-means++ or 1d')
    args = parser.parse_args()
    if args.task == 'austin_reviews':
        # business_data = 
        austin_businesses = GetBusinessIDCsv()

        austin_reviews = GetReviewCSV(austin_businesses)
    elif args.task == 'optimal_k' or args.task == 'review_clustering':
        businesses = []
        coordinates = []
        reviews = []
        with open('./austin_reviews.csv') as rfile:
            for line in rfile:
                info = line.replace('[','').replace(']','').replace("'",'').split(';')
                businesses.append(info[0])
                reviews.append(info[1])

        with open('./austin_businesses.csv') as file:
            for line in file:
                info = line.split(';')
                latitude = float(info[1]) % 1
                longitude = float(info[2]) %1

                coordinate = [latitude, longitude]
                coordinates.append(coordinate)
        cdata = np.array(coordinates)
        data = np.array(reviews, dtype='<U100000')
        tfidf = TfidfVectorizer()
        tfidf.fit(data)
        X = tfidf.transform(data)
        if args.task == 'optimal_k':
            distances = []
            K = range(2,10)
            for k in K:
                km = KMeans(n_clusters=k, max_iter=300, n_init = 10)
                km = km.fit(X)
                distances.append(km.inertia_)

            plt.plot(K, distances)
            plt.xlabel('k')
            plt.ylabel('sum of distances')
            plt.show()
        else:
            true_k = 6
            km = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=10)
            km.fit(X)
            labels = km.predict(X)

            df=pd.DataFrame(list(zip(businesses,labels)),columns=['business','cluster'])
            df = df.sort_values(by='business')
            df.to_csv('./review_cluster.csv')
            colormap = np.array(['red','blue','green','black','cyan','orange'])
            plt.scatter(cdata[:,0], cdata[:,1], c=colormap[labels])
            plt.xlabel('latitude')
            plt.ylabel('longitude')
            plt.title('review clustering')
            plt.show()
    
    elif args.task == 'coordinate_clustering':
        coordinates = []
        businesses = []
        with open('./austin_businesses.csv') as file:
            for line in file:
                info = line.split(';')
                businesses.append(info[0])
                latitude = float(info[1]) % 1
                longitude = float(info[2]) %1

                coordinate = [latitude, longitude]
                coordinates.append(coordinate)
        
        data = np.array(coordinates)
        
        true_k = 6
        km = KMeans(n_clusters=true_k, init='k-means++')
        km.fit(data)

        cluster_label = km.fit_predict(data)
        labels = km.predict(data)

        df=pd.DataFrame(list(zip(businesses,labels)),columns=['business','cluster'])
        # df = df.sort_values(by='business')
        df.to_csv('./coordinate_cluster.csv')

        colormap = np.array(['red','blue','green','black','cyan','orange'])
        plt.scatter(data[:,0], data[:,1], c=colormap[labels])
        plt.xlabel('latitude')
        plt.ylabel('longitude')
        plt.title('coordinate clustering')
        plt.show()
        pass


    elif args.task == 'category_clustering':
        categories = []
        businesses = []
        with open('./austin_businesses.csv') as file:
            for line in file:
                info = line.split(';')
                businesses.append(info[0])
                categories.append(info[3])
        coordinates = []
        with open('./austin_businesses.csv') as file:
            for line in file:
                info = line.split(';')
                latitude = float(info[1]) % 1
                longitude = float(info[2]) %1

                coordinate = [latitude, longitude]
                coordinates.append(coordinate)
        cdata = np.array(coordinates)
        data = np.array(categories, dtype='U')
        tfidf = TfidfVectorizer()
        tfidf.fit(data)
        X = tfidf.transform(data)
        true_k = 6
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=10)
        km.fit(X)
        labels = km.predict(X)

        df=pd.DataFrame(list(zip(businesses,labels)),columns=['business','cluster'])
        df = df.sort_values(by='business')
        df.to_csv('./categories_cluster.csv')
        colormap = np.array(['red','blue','green','black','cyan','orange'])
        plt.scatter(cdata[:,0],cdata[:,1], c=colormap[labels])
        plt.xlabel('latitude')
        plt.ylabel('longitude')
        plt.title('category clustering')
        plt.show()
        pass
    elif args.task == 'predict':
        businesses = {}
        reviews = []
        
        with open('./austin_businesses.csv') as file:
            for line in file:
                info = line.split(';')
                bid = info[0]
                if bid not in businesses:
                    businesses[bid] = info

        with open(REVIEW_FILE) as rfile:
            for line in rfile:
                data = json.loads(line)
                bid = data.get('business_id')
                if bid in businesses:
                    review_id = data.get('review_id')
                    stars = data.get('stars')
                    text = data.get('text')
                    review = [review_id, stars, text]
                    reviews.append(review)
            
        print("Finished reading files")
        review_data = np.array(reviews, dtype='<U1000')
        data_sample = review_data[np.random.randint(review_data.shape[0], size = 100000)]
        print(data_sample.shape)
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf.fit(data_sample[:,2])
        X = tfidf.transform(data_sample[:,2])
        y = data_sample[:,1]

        

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=42)
        knn = neighbors.KNeighborsClassifier(n_neighbors=1)
        pred = knn.fit(X_train, y_train).predict(X_test)
        print(classification_report(y_test, pred, digits=3))
        pass
    # GetReviewCSV(REVIEW_FILE, restaurants)
    
    pass