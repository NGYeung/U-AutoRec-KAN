# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 02:17:01 2024

@author: Yiyangl
"""


import numpy as np
import torch
import torch.nn as nn
import math


from AutoEncoders import DAE, DAE_KAN





class MatrixFactorization_AE(nn.Module):
    '''
    A "Matrix Factorization type" CF filtering recommender method using VAE
    '''
    
    def __init__(self, user_feature_size, item_feature_size, num_item, user_batch_size = 64, 
                 AE_hidden_dimension = 256, embedding_dimension = 24, autoencoder = 'DAE', noise = 0.05 ):
        '''
        
        Parameters
        ----------
        user_feature_size : int
            the size of each user feature
        item_feature_size : int
            the size of each item feature
        num_item : TYPE
            the number of items in the database
        user_batch_size : int, optional
            the batch size of user input. The default is 128.
        AE_hidden_dimension : TYPE, optional
            hidden dimension for the autoencoder. The default is 16.
        embedding_dimension : TYPE, optional
            dimension of the user/item embedding. The default is 48.
        autoencoder : TYPE, optional
            which autoencoder to use. The default is 'VAE'. 
            Available options: 
            
        Raises
        ------
        ValueError
            Wrong autoencoder option

        Returns
        -------
        None.

        '''
        

        super(MatrixFactorization_AE, self).__init__()
        self.ae = autoencoder
        
        print("\n \t ----------- Model = Recommender with Autoencoder ------------")
        self.user_projection = nn.Sequential(
            nn.Linear(user_feature_size, embedding_dimension),
            nn.ReLU(), #ReLU or SeLU
            nn.Dropout(0.1),
            nn.LayerNorm(embedding_dimension)
            )
        
        
        self.item_projection = nn.Sequential(
            nn.Linear(item_feature_size, embedding_dimension),
            nn.ReLU(), #ReLU or SeLU
            nn.Dropout(0.1),
            nn.LayerNorm(embedding_dimension)
            )
        
        # size: user_batch_size x num_item
        self.CF_matrix = None
        self.target_matrix = None
        
        print(f"\n \t \t Autoencoder = {autoencoder} ")
        if autoencoder == 'VAE':
            self.AE = VAE(num_item, AE_hidden_dimension)
            
        elif autoencoder == 'DAE':
            self.AE = DAE(num_item, AE_hidden_dimension, noise = noise)
 
        elif autoencoder == 'DAE_KAN':
        
            self.AE = DAE_KAN(num_item, AE_hidden_dimension, noise = noise)
        
        else:
            raise ValueError('autoencoder = VAE, DAE, or DAE_KAN')
        torch.autograd.set_detect_anomaly(True)
            

            
    def forward(self, user_feature, item_feature, rating_range = 5):
        
        user = self.user_projection(user_feature.unsqueeze(1)) # user_batch_size x 1 x embedding_dim
        item = self.item_projection(item_feature.unsqueeze(1)) # num_items x 1x embedding_dim
        #print(user.shape)
        #print(item.shape)
        self.CF_matrix = user.squeeze(1) @ item.squeeze(1).T # user_batch_size x num_items
        
        # Normalize the CF matrix to range between 0 and 1
        self.CF_matrix = (self.CF_matrix - self.CF_matrix.min()) / (self.CF_matrix.max() - self.CF_matrix.min())
        
        self.CF_matrix = self.CF_matrix.unsqueeze(1) #batch_size x 1 x items
        
        
        if self.ae == "VAE":
            Reconstruction,mean, log_var = self.AE(self.CF_matrix)
            if torch.isnan(Reconstruction).any():
                print("NaNs found in Reconstruction")
            
  
            # training with MSE or CrossEntropy
        
        
        
            return Reconstruction, mean, log_var
        
        if self.ae =='DAE' or self.ae == 'DAE_KAN':
            Reconstruction,_, _ = self.AE(self.CF_matrix)
            if torch.isnan(Reconstruction).any():
                print("NaNs found in Reconstruction")
            
  
            # training with MSE or CrossEntropy
        
        
        
            return Reconstruction, None, None
        
        
        
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        