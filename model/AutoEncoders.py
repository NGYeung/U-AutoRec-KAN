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
'''
git clone https://github.com/ZiyaoLi/fast-kan
cd fast-kan
pip install .

'''

# or KAN implementation






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
        self.mu = nn.Parameter(torch.randn(self.hidden))
        self.noise_strength = noise
        
        self.encoder = nn.Sequential(
            nn.Linear(matrix_size, self.hidden),
            #nn.SELU(), # Parameter to tune: the size of hidden layer (Rank of CF??)
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden, matrix_size),
            #nn.SELU(),
            nn.Sigmoid() #sigmoid for non-exclusive outputs
            )
        self.b = nn.Parameter(torch.randn(matrix_size))


    
    def forward(self, da_CF_matrix):
        
    
        gaussian_noise = torch.rand_like(da_CF_matrix) #create the gaussian noise. std = 1 mean = 0

        noise = gaussian_noise*self.noise_strength
        hidden_rep = self.encoder(da_CF_matrix + noise) + self.mu
        decoded  = self.decoder(hidden_rep) *5 + self.b
        
        return decoded, None, None # return both
        
    
    

        
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
            nn.Linear(matrix_size, self.hidden),
            nn.SELU(), # Parameter to tune: the size of hidden layer (Rank of CF??)
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden, matrix_size),
            nn.SELU(),
            nn.Sigmoid() #sigmoid for non-exclusive outputs
            )
        
        nn.init.kaiming_uniform_(self.mu.weight, nonlinearity='relu')
        if self.mu.bias is not None:
            nn.init.constant_(self.mu.bias, 0)

        nn.init.kaiming_uniform_(self.log_var.weight, nonlinearity='relu')
        if self.log_var.bias is not None:
            nn.init.constant_(self.log_var.bias, 0)


    def reparameterize(self, mu, log_var):
        
        # compute standard deviation from log variance
        
        if torch.isnan(mu).any() or torch.isnan(log_var).any():
            print("NaNs found in mean or log_var!")
        
        # Adding clipping to log_var to avoid extreme values
        log_var = torch.clamp(log_var, min=-5, max=5)
        
        std = torch.exp(log_var/2)
        sample = torch.rand_like(std) # create a random matrix with the same shape with std = 1
    
        return sample*std + mu
        
        
    def forward(self, da_CF_matrix):
        

        hidden = self.encoder(da_CF_matrix)

        mean = self.mu(hidden)
        log_var = self.log_var(hidden)
        sample = self.reparameterize(mean, log_var)
        decoded  = self.decoder(sample)*5

        return decoded, mean, log_var
    



#KAN based Variational VAE.
class VAE_KAN(nn.Module):
    
    def __init__(self, matrix_size, hidden_size):
        
        
        super(VAE_KAN, self).__init__()
        
        self.hidden = hidden_size
        
        # The log variance and the mean of the hidden distribution
        self.mu = FastKANLayer(self.hidden, self.hidden)
        self.log_var = FastKANLayer(self.hidden, self.hidden)
        
        
        self.encoder = nn.Sequential(
            FastKANLayer(matrix_size, self.hidden), # Parameter to tune: the size of hidden layer (Rank of CF??)
            )
        
        self.decoder = nn.Sequential(
            FastKANLayer(self.hidden, matrix_size),
            nn.Sigmoid() #sigmoid for non-exclusive outputs
            )
        
        
    def reparameterize(self, mu, log_var):
            
        # compute standard deviation from log variance
        
        if torch.isnan(mu).any() or torch.isnan(log_var).any():
            print("NaNs found in mean or log_var!")
        
        # Adding clipping to log_var to avoid extreme values
        log_var = torch.clamp(log_var, min=-5, max=5)
        
        std = torch.exp(log_var/2)
        sample = torch.rand_like(std) # create a random matrix with the same shape with std = 1
    
        return sample*std + mu
    
    
    def forward(self, da_CF_matrix):
        

        hidden = self.encoder(da_CF_matrix)

        mean = self.mu(hidden)
        log_var = self.log_var(hidden)
        sample = self.reparameterize(mean, log_var)
        decoded  = self.decoder(sample)*5

        return decoded, mean, log_var
    
    
        


class DAE_KAN(nn.Module):
    
    
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
        
        
        super(DAE_KAN, self).__init__()
        
        self.hidden = hidden_size
        
        self.noise_strength = noise
        
        self.encoder = nn.Sequential(
            FastKANLayer(matrix_size, self.hidden),
            )
        
        self.decoder = nn.Sequential(
            FastKANLayer(self.hidden, matrix_size),
            nn.Sigmoid() #sigmoid for non-exclusive outputs
            )


    
    def forward(self, da_CF_matrix):
        
    
        gaussian_noise = torch.rand_like(da_CF_matrix) #create the gaussian noise. std = 1 mean = 0

        noise = gaussian_noise*self.noise_strength
        hidden_rep = self.encoder(da_CF_matrix + noise)
        decoded  = self.decoder(hidden_rep) *5
        
        return decoded, None, None # return both