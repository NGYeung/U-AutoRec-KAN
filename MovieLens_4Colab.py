
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

# Replace with your own pre-processed data.
filepath_1m ={'user': r"/content/drive/MyDrive/RecSys/ml-1m/users.csv",
              'rating': r"/content/drive/MyDrive/RecSys/ml-1m/ratings.csv",
              'movie': r"/content/drive/MyDrive/RecSys/ml-1m/movies.csv"}

filepath = {
    'user': r"/content/drive/MyDrive/RecSys/ml-100k/User_EVERYTHING.csv",
    'rating': r"/content/drive/MyDrive/RecSys/ml-100k/rating_info.csv",
    'movie': r"/content/drive/MyDrive/RecSys/ml-100k/movies_info.csv"}



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
      
        
        # the rating matrix
        for i, data in enumerate(self.data.values):
            # keys: user_id:0 age: 1,  zip_code: 2, occupation: 3, movie_id: 4. genre:5, rating:6
            if i in training_set_indicies:
                #print(data)
                train_rating[int(data[0])-1,int(data[4])-1] = int(data[6])
                train_mask[int(data[0])-1,int(data[4])-1]  = 1
                
            else:
                test_rating[int(data[0])-1,int(data[4])-1] = int(data[6])
                test_mask[int(data[0])-1,int(data[4])-1] = 1
                
                
    
        return train_rating, train_mask, test_rating, test_mask
    
    
    
    



class Movie_100K():
    '''
    The movie lens 100K dataset for the reinforcement learning playground
    '''
    
    
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
    
        self.item.pop('item_id')
        self.item.pop('Unnamed: 0')
 
        if self.need_embedding:
            self.item['title_embedding'] = self.embedding[idx,:]

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

        self.items = pd.read_csv(self.path['movie'])
 
        self.items = self.items.apply(self.process_movie, axis = 1)
        self.items = self.items[['movie_id','date','genre']]
        self.item_num = self.items.shape[0]

        ratings = pd.read_csv(self.path['rating'])
  
   
        big_table = pd.merge(self.users, ratings, left_on='user_id', right_on='user_id')
        big_table = pd.merge(big_table, self.items, left_on='item_id', right_on='movie_id')
        
        self.data = big_table
    
        if self.need_embedding:
            self.embedding = torch.load(self.path['embedding'])

        self.items = self.items[['movie_id','date','genre']]

    
    
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
    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2), "Datasets must be of the same length"
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        data1 = self.dataset1[idx]
        data2 = self.dataset2[idx]
   
        return data1, data2
        
        
def pre_process_100K(train, test, user_num=943, item_num=1682):
    '''
    spit out the rating matrices and the masks
    '''
    
    train_rating = np.zeros((user_num, item_num))
    test_rating = np.zeros((user_num, item_num))
    train_mask = np.zeros((user_num, item_num))
    test_mask = np.zeros((user_num, item_num))
      
        
        # the rating matrix
        for i, data in enumerate(train):
            # keys: user_id:0 age: 1,  zip_code: 2, occupation: 3, movie_id: 4. genre:5, rating:6

            train_rating[int(data[0])-1,int(data[1])-1] = int(data[2])
            train_mask[int(data[0])-1,int(data[1])-1]  = 1
            
        for j, data in enumerate(test):
                
            test_rating[int(data[0])-1,int(data[1])-1] = int(data[2])
            test_mask[int(data[0])-1,int(data[1])-1] = 1

    return train_rating, train_mask, test_rating, test_mask 
