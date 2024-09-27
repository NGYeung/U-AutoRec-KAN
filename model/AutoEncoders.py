# -*- coding: utf-8 -*-
"""
A collections of autoencoder models for future use.

Created on Thu Sep 26 18:12:11 2024

@author: Yiyang Liu
"""

import numpy as np

import torch
import torch.nn as nn
import math

from fastkan import FastKANLayer


# or KAN implementation
'''
git clone https://github.com/ZiyaoLi/fast-kan
cd fast-kan
pip install .

'''





# Denoiseing Autoencoder to reconstruct the CF interaction matrix given known ratings.
class DAE(nn.Module):
    
    
    def __init__(self, matrix_size, hidden_size, noise = 0.05):
        ''' 
        Parameters
        ----------
        matrix_size : int
            the size of the CF matrix (Flattened)
           
        hidden_size : int
            the size of hidden representation
            (recommendation: calculate with SVD)
            
        noise: float
            the strength of the noise add to the input. Default = 0.05
            if set to zero then it's a regular autoencoder.

        Returns
        -------
        None.

        '''
        
        
        super(DAE, self).__init__()
        
        self.hidden = hidden_size
        
        self.noise_strength = noise
        
        self.encoder = nn.Sequential(
            nn.Linear(matrix_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden),
            nn.ReLU(), # Parameter to tune: the size of hidden layer (Rank of CF??)
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, matrix_size),
            nn.ReLU(),
            nn.Sigmoid() #sigmoid for non-exclusive outputs
            )


    
    def forward(self, da_CF_matrix):
        
    
        gaussian_noise = torch.rand_like(da_CF_matrix) #create the gaussian noise. std = 1 mean = 0

        noise = gaussian_noise*self.noise_strength
        hidden_rep = self.encoder(da_CF_matrix + noise)
        decoded  = self.decoder(hidden_rep) 
        
        return decoded, hidden_rep # return both
        
    
    

        
# Variational Autoencoder        
class VAE(nn.Module):
    
    
    def __init__(self, matrix_size, hidden_size):
        '''
        

        Parameters
        ----------
        matrix_size : int
            the size of the CF matrix (Flattened)
           
        hidden_size : int
            the size of hidden representation
            (recommendation: calculate with SVD)

        Returns
        -------
        None.

        '''
        
        super(VAE, self).__init__()
        
        self.hidden = hidden_size
        
        # The log variance and the mean of the hidden distribution
        self.mu = nn.Linear(self.hidden, self.hidden)
        self.log_var = nn.Linear(self.hidden, self.hidden)
        
        
        self.encoder = nn.Sequential(
            nn.Linear(matrix_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden),
            nn.ReLU(), # Parameter to tune: the size of hidden layer (Rank of CF??)
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, matrix_size),
            nn.ReLU(),
            nn.Sigmoid() #sigmoid for non-exclusive outputs
            )
        
        nn.init.kaiming_uniform_(self.mu, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.log_var.weight, nonlinearity='relu')


    def reparameterize(self, mu, log_var):
        
        # compute standard deviation from log variance
        std = torch.exp(log_var/2)
        sample = torch.rand_like(std) # create a random matrix with the same shape with std = 1
    
        return sample*std + mu
        
        
    def forward(self, da_CF_matrix):
        

        hidden_rep = self.encoder(da_CF_matrix)
        mean = self.mu(hidden_rep)
        log_var = self.log_var(hidden_rep)
        sample = self.reparameterize(mean, log_var)
        decoded  = self.decoder(sample)
        
        return decoded, hidden_rep, mean, log_var
    



#KAN based Variational VAE.
class VAE_KAN(nn.Module):
    
    def __init__(self, matrix_size, hidden_size):
        
        
        super(VAE_KAN, self).__init__()
        
        self.hidden = hidden_size
        
        # The log variance and the mean of the hidden distribution
        self.mu = FastKANLayer(self.hidden, self.hidden)
        self.log_var = FastKANLayer(self.hidden, self.hidden)
        
        
        self.encoder = nn.Sequential(
            FastKANLayer(matrix_size, 128),
            FastKANLayer(128, self.hidden), # Parameter to tune: the size of hidden layer (Rank of CF??)
            )
        
        self.decoder = nn.Sequential(
            FastKANLayer(self.hidden_size, 128),
            FastKANLayer(128, matrix_size),
            nn.Sigmoid() #sigmoid for non-exclusive outputs
            )
        
        
    def reparameterize(self, mu, log_var):
            
        # compute standard deviation from log variance
        std = torch.exp(log_var/2)
        sample = torch.rand_like(std) # create a random matrix with the same shape with std = 1
        
        return sample*std + mu
    
    
    
    def forward(self, da_CF_matrix):
        

        hidden_rep = self.encoder(da_CF_matrix)
        mean = self.mu(hidden_rep)
        log_var = self.log_var(hidden_rep)
        sample = self.reparameterize(mean, log_var)
        decoded  = self.decoder(sample)
        
        return decoded, hidden_rep, mean, log_var
    
        


