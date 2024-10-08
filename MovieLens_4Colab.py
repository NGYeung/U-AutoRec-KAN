# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 19:20:16 2024

@author: Yiyang Liu

The Dataset Class for the class, to help with the training
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from transformers import BertTokenizer
from torch.utils.data import DataLoader
#import spacy

filepath_1m ={'user': r"/content/drive/MyDrive/RecSys/ml-1m/users.csv",
              'rating': r"/content/drive/MyDrive/RecSys/ml-1m/ratings.csv",
              'movie': r"/content/drive/MyDrive/RecSys/ml-1m/movies.csv"}

filepath = {
    'user': r"/content/drive/MyDrive/RecSys/ml-100k/User_EVERYTHING.csv",
    'rating': r"/content/drive/MyDrive/RecSys/ml-100k/rating_info.csv",
    'movie': r"/content/drive/MyDrive/RecSys/ml-100k/movies_info.csv",
    'embedding': r"/content/drive/MyDrive/RecSys/ml-100k/encoded_text_dim16.pt"}




class Movie_1M():
    
    def __init__(self, filename = filepath_1m):
        
        self.path = filename
        self.users = None
        self.items = None
        self.data = None #The file with the ratings
        self.load()
        
    
    def load(self):
        
        self.users = pd.read_csv(self.path['user'])
        self.user_num = 6040
        self.data = pd.read_csv(self.path['rating'])
        self.item_num = 3952
        self.items = pd.read_csv(self.path['movie']) 
        
        self.users = self.users[['user_id', 'age', 'gender','zip_code','occupation']]
        self.users['user_id'] = self.users['user_id'].apply(int)

        self.users['age'] = self.users['age'].apply(eval)
        self.users['zip_code'] = self.users['zip_code'].apply(eval)
        self.users['occupation'] = self.users['occupation'].apply(eval)
        self.items = self.items[['movie_id','genre']]
        self.items['movie_id'] = self.items['movie_id'].apply(int)
        self.items['genre'] = self.items['genre'].apply(eval)
        
        big_table = pd.merge(self.users, self.data, left_on='user_id', right_on='user_id')
        big_table = pd.merge(big_table, self.items, left_on='movie_id', right_on='movie_id')
        self.data = big_table
        self.data = self.data[['user_id','age','zip_code','occupation', 'movie_id', 'genre', 'rating']]

    
    
    def __len__(self):
        
        return len(self.data)

    
    
    def __getitem__(self, idx):
        
        data = self.data[['user_id','age','zip_code','occupation', 'movie_id', 'genre', 'rating']]
        
        return data.iloc[idx]
    
    
    
    def preprocessor(self, training_set_indicies, fourier = True):
        
        train_rating = np.zeros((self.user_num, self.item_num))
        test_rating = np.zeros((self.user_num, self.item_num))
        train_mask = np.zeros((self.user_num, self.item_num))
        test_mask = np.zeros((self.user_num, self.item_num))
        user_matrix = np.zeros((self.user_num, 39)) #remember the calculate and change the number
        if fourier:
            user_matrix = np.zeros((self.user_num, 67)) 
            
        item_matrix = np.zeros((self.item_num, 19)) #remember to calculate and change the number
        if fourier:
            item_matrix = np.zeros((self.item_num, 21)) 
            
        user_genre_ct = np.zeros((self.user_num, 18))
        user_genre_rt = np.zeros((self.user_num, 18))
        #item_genre = np.zeros((fullset.user_num, 20))
        #user_genre = np.zeros((self.user_num, 20))
        movie_avg = np.zeros((self.item_num,))
        movie_ct = np.zeros((self.item_num,))
        
        
        # the rating matrix
        for i, data in enumerate(self.data.values):
            # keys: user_id:0 age: 1,  zip_code: 2, occupation: 3, movie_id: 4. genre:5, rating:6
            if i in training_set_indicies:
                #print(data)
                train_rating[int(data[0])-1,int(data[4])-1] = int(data[6])
                train_mask[int(data[0])-1,int(data[4])-1]  = 1
                user_genre_ct[int(data[0])-1,:] += np.array(data[5])
                user_genre_rt[int(data[0])-1,:] += int(data[6])*np.array(data[5])
                movie_ct[int(data[4])-1] += 1
                movie_avg[int(data[4])-1] += int(data[6])
                
            else:
                test_rating[int(data[0])-1,int(data[4])-1] = int(data[6])
                test_mask[int(data[0])-1,int(data[4])-1] = 1
                
                
        
        user_genre_rt = user_genre_rt / (user_genre_ct+1e-12) #user's average score for each genre
        movie_avg /= (movie_ct + 1e-12) # average rating for each movie  (in the training set)
    
        
        for i, user in enumerate(self.users.values): 
            
            
            if not fourier: # age: 7 gender: 1 location: 11 occupation: 20 + genre = 18
                user_matrix[i,:39] = np.concatenate([np.atleast_1d(item).ravel() for item in user])[1:] 
                #user_matrix[i,39:] = user_genre_rt[i]
                
            else:

                user_matrix[0] = user[2]
                # age
                bases = fourier_bases(7, 10)
                age = user[1]
                age_ = np.zeros((10,))
                for j, a in enumerate(age):
                    age_ += a*bases[j]
                user_matrix[i,1:11] = age_
                

                #location
                bases = fourier_bases(11, 15)
                location = user[3]
                location_ = np.zeros((15,))
                for j, z in enumerate(location):
                    location_ += z*bases[j]
                    
                user_matrix[i, 12:27] = location_
                
                #occupation
                bases = fourier_bases(20, 20)
                occupation = user[4]
                occupation_ = np.zeros((20,))
                for j, z in enumerate(occupation):
                    occupation_ += z*bases[j]
                    
                user_matrix[i, 27:47] = occupation_
                
                #genre_rating
                bases = fourier_bases(18, 20)
                genre = np.zeros((20,))
                for j, z in enumerate(user_genre_rt[i]):
                    genre += z*bases[j]
                    
                #user_matrix[i, 47:] = genre
                
        for k, item in enumerate(self.items.values):
            n = item[0]
            
            if not fourier:
                
                item_matrix[n-1,:18] = np.concatenate([np.atleast_1d(i).ravel() for i in item])[1:]
                item_matrix[n-1,18] = movie_avg[n-1]
                
                
                
            else:
                bases = fourier_bases(18, 20)
                genre = np.zeros((20, ),dtype='float64')
                
                for j, z in enumerate(item[1]):
                    genre += z*bases[j]
                
                    
                item_matrix[n-1,:20] = genre
                    
                item_matrix[n-1,20] = movie_avg[n-1]

        return train_rating, train_mask, test_rating, test_mask, user_matrix, item_matrix
    
    
    
    



class Movie_100K():
    
    def __init__(self, filename = filepath, for_training = False, embedding = False):
        '''
        Input: filename[dict] = {rating_file:path, user_file:path, movie_file:path}

        '''
        
        self.encode_title = 0 
        self.user_num = 0
        self.item_num = 0
        self.need_embedding = embedding
        self.item = {}
        self.path = filename
        self.load()
        self.text_tokenizer = None
        if for_training:
            self.encode_title = 1
            #self.text_tokenizer = BertTokenizer('' )

        
        
    def __len__(self):
        
        return self.data.shape[0]
    
        
    
    def __getitem__(self, idx):
        '''
        Input: the index
        return items: dictionary
        {'timestamp':[Int] the time-stamp of the recommendation, 
         'user':[Dict] a dictionary of user profile,
         'movie':[Dict] Information of the rated movie. }
        
        movie:{'movieid':[Int], 'title':[Str], 'date':[Int], 'genre':nparray }
        user:{'user_id':[Int], 'age':[Int], gender:[Int] 'occupation':nparray}
        '''
        self.item = self.data.iloc[idx,:].to_dict()
        #self.item.pop('item_id_y')
        self.item.pop('item_id')
        self.item.pop('Unnamed: 0')
        #self.item.pop('Unnamed: 0_y')
        if self.need_embedding:
            self.item['title_embedding'] = self.embedding[idx,:]
        #orient='records' exclude the index
        return self.item
    
    
    def load(self):
        '''
        Load Data and Process
        1. convert the occupation to one-hot
        2. DateTime to timestamp for the movie date
        3. Gender to boolean.
        '''
        self.users = pd.read_csv(self.path['user'])
        
        self.users = self.users.apply(self.process_user, axis = 1)
        self.users = self.users[['user_id', 'age', 'gender','zip_code','occupation']]
        self.user_num = self.users.shape[0]
        #print(users.iloc[1])
        self.items = pd.read_csv(self.path['movie'])
        #print(movies.iloc[1])
        self.items = self.items.apply(self.process_movie, axis = 1)
        self.items = self.items[['movie_id','date','genre']]
        self.item_num = self.items.shape[0]
        #print(movies.iloc[1])
        ratings = pd.read_csv(self.path['rating'])
        #movie_avg = ratings[['item_id','rating']].groupby('item_id').mean().reset_index()
        #movie_avg.columns = ['item_id','film_avg_rating']
        #self.items = pd.merge(self.items, movie_avg, left_on='movie_id', right_on='item_id')
        
        #self.items = self.items.drop('item_id')
   
        
        
        # merge all into a big table
        big_table = pd.merge(self.users, ratings, left_on='user_id', right_on='user_id')
        big_table = pd.merge(big_table, self.items, left_on='item_id', right_on='movie_id')
        
        self.data = big_table
    
        if self.need_embedding:
            self.embedding = torch.load(self.path['embedding'])

        self.items = self.items[['movie_id','date','genre']]
        #self.items['title_embedding'] = self.embedding
        
        
    
    
    
    def process_movie(self,row):
        '''
        Encode the dates of the movie
        input format: Jan-01-1995
        '''
        
        date = pd.to_datetime(row['date'])
        
        year = date.year
        mon = date.month
        day = date.day
        #print(year, mon, day)
        dt_vec = []
        dt_vec.append((2000-year)/5)
        #month encoding
        dt_vec += [np.sin(2 * np.pi * mon/ 12), np.cos(2 * np.pi * mon / 12)]
        dt_vec += [np.sin(2 * np.pi * day/ 12), np.cos(2 * np.pi * day / 31)]
        
        row['date'] = np.array(dt_vec)
        
      
        
        g = row['genre']
        g = g[1:len(g)-1].split()
        g = [int(i) for i in g]
        row['genre'] = np.array(g)
        '''
        rep = np.zeros((20,))
        bases = fourier_bases(19, 20)
        for i in range(len(g)):
            rep += g[i]*bases[i]
        
        row['genre'] = rep/np.linalg.norm(rep)
        '''
        return row[['movie_id','title','date','genre']]
       

    def process_user(self,row):

        occupation = list(row.iloc[24:])
        
        rep = np.zeros((25,))
        bases = fourier_bases(20, 25)
        for i in range(len(occupation)):

            rep += np.array(occupation[i]*bases[i])
        
        
        rep = rep/(np.linalg.norm(rep)+1e-15)
        
        
        average_rating = list(row.iloc[4:24])
        zip_code = [0 for i in range(11)]
        #print(occupation,average_rating)
        
        upperbound = [18,25,35,45,50,56]
        age = [0 for i in range(7)]
        i = 0
        while i < 6 and upperbound[i] <= row['age']:
            i += 1
        age[i] = 1
            
        row['age'] = age
        
        row['occupation'] = rep
        row['average_rating'] = average_rating
        
        bases = fourier_bases(11,12)
        
        for z in row['zip_code']:
            canada = False
            for i in z: 
                if i in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    canada = True
        
            if canada:
                zip_code = bases[10]
            else:
                zip_code = bases[int(z[0])]
            
        row['zip_code'] = zip_code
         #change to select only the infos.
        
        return row[['user_id','age','gender','zip_code','occupation']]


    
def Data2Embedding(Data):
    '''
    A function to map the data to states in RL
    Input: one instance of Data
    '''
    stack = []
    stack.append(Data['average_rating'][-1])
    stack.append(Data['film_avg_rating'])
    stack += Data['title_embedding'].tolist()
    stack += list(Data['date'])
    stack += list(Data['genre'])
    stack.append(Data['age'])
    stack.append(Data['gender'])
    stack += torch.tensor(Data['occupation']).tolist()
    
    
    # An observation includes:
    # observation 0: the average rating of the film and the average rating given by the user
    # observation 1: title_embedding. Choose embedding length = 8
    # observation 2: move date 
    # observation 3: movie genre. = MultiDiscrete([2]*19)
    # observation 4: user age
    # observation 5: user gender
    # observation 6: user occupation
    # observation 7: user location 
    #so far location is excluded, can be added in the future. 
    
    
    return np.array(stack)
    # use np array because we want to use njit for state operation.


def fourier_bases(frequency_count, n_points):
    """
    Generate the sine Fourier bases at n points from 0 to pi.
    
    Parameters:
    frequency_count (int): The number of sine Fourier bases to generate (frequency up to frequency_count).
    n_points (int): The number of points from 0 to pi at which to evaluate the sine Fourier bases.
    
    Returns:
    bases (ndarray): An array of shape (frequency_count, n_points), where each row represents a sine Fourier basis.
    """
    x_points = np.linspace(0, np.pi, n_points, endpoint=False)
    bases = []
    
    # Generate sine bases
    for k in range(1, frequency_count + 1):
        base = np.sin(k * x_points)
        bases.append(base)
    
    return np.array(bases)
    
    
class CombinedDataset:
    def __init__(self, dataset1, dataset2, dataset3):
        assert len(dataset1) == len(dataset2), "Datasets must be of the same length"
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        data1 = self.dataset1[idx]
        data2 = self.dataset2[idx]
        data3 = self.dataset3[idx]
        return data1, data2, data3