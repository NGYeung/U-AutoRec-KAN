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
#import spacy
from numba import jit




filepath = {
    'user': r"C:\Users\yyBee\Datasets\ml-100k\User_EVERYTHING.csv",
    'rating': r"C:\Users\yyBee\Datasets\ml-100k\rating_info.csv",
    'movie': r"C:\Users\yyBee\Datasets\ml-100k\movies_info.csv",
    'embedding': r"C:\Users\yyBee\Datasets\ml-100k\encoded_text_dim16.pt"}

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
        #self.data = self.data.apply(self.encode_user, axis = 1)
        #self.data = self.data.apply(self.encoder_dates, axis = 1)
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
        
        row['age'] = row['age']/30
        
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


    
@jit
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

    
    
class CombinedDataset(Dataset):
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
    
        
        
    
        
        
        
        