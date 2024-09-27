# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 02:17:01 2024

@author: Yiyangl
"""


import numpy as np
import torch
import torch.nn as nn
import math

from AutoEncoders import VAE, VAE_KAN, DAE





class MatrixFactorization_VAE(nn.Module):
    '''
    A "Matrix Factorization type" CF filtering recommender method using VAE
    '''
    
    def __init__(self, user_feature_size, item_feature_size, num_item, user_batch_size = 128, 
                 AE_hidden_dimension = 16, embedding_dimension = 48, autoencoder = 'VAE' ):
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
        
        
        print("\n \t ----------- Model = Recommender with Autoencoder ------------")
        self.user_projection = nn.Sequential(
            nn.Linear(user_feature_size, 96),
            nn.ReLU(), #ReLU or SeLU
            nn.Linear(96, embedding_dimension),
            nn.Dropout(0.1),
            nn.LayerNorm()
            )
        
        
        self.item_projection = nn.Sequentialnn.Sequential(
            nn.Linear(user_feature_size, 96),
            nn.ReLU(), #ReLU or SeLU
            nn.Linear(96, embedding_dimension),
            nn.Dropout(0.1),
            nn.LayerNorm()
            )
        
        # size: user_batch_size x num_item
        self.CF_matrix = None
        self.target_matrix = None
        
        print(f"\n \t \t Autoencoder = {autoencoder} ")
        if autoencoder == 'VAE':
            self.AE = VAE(num_item*user_batch_size, AE_hidden_dimension)
            
        elif autoencoder == 'DAE':
            self.AE = DAE(num_item*user_batch_size, AE_hidden_dimension)
 
        elif autoencoder == 'VAE_KAN':
            self.AE = VAE_KAN(num_item*user_batch_size, AE_hidden_dimension)
        
        else:
            raise ValueError('autoencoder = VAE, DAE, or VAE_KAN')
            

            
    def forward(self, user_feature, item_feature, rating_range = 5):
        
        user = self.user_projection(user_feature) # user_batch_size x embedding_dim
        item = self.item_projection(item_feature) # num_items x embedding_dim
        
        self.CF_matrix = user @ item # user_batch_size x num_items
        self.CF_matrix = self.CF_matrix.view(-1) # FLATTEN IT!
        
        
        Reconstruction = rating_range*self.AE(self.CF_matrix)
        # training with MSE or CrossEntropy
        
        return Reconstruction
        
        
        
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        