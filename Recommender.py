# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 23:13:26 2024

@author: Yiyang Liu
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import math
import time
from numba import jit

from model.Matrix_models import MatrixFactorization_VAE
from torch.utils.data import DataLoader, Subset 
import yaml



class AEReco:
    
    def __init__(self, configuration_file):
        
        

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.config(configuration_file)
        #self.dataset = DataLoader(data) 
        user_feature_size = self.user_feature_size
        item_feature_size = self.item_feature_size
        num_item = self.num_items
        batch = self.batch_size
        
        
        self.model = MatrixFactorization_VAE(user_feature_size, item_feature_size, num_item, batch_size = batch, AE_hidden_dimension = self.AE_bottleneck).to(self.device)
        #self.model = nn.DataParallel(self.model,device_ids=[0,1,2,3])
        self.target = None
        self.user = None
        self.film = None
        self.optimizer = optim.ASGD(self.model.parameters, lr=self.learning_rate).to(self.device) #add learning rate and other options
        self.lossfn = nn.MSELoss()
        self.loss_log = []
        
        
        print("\n \t ----------- Model Loaded ------------")
        print("\t *Total Params* = ",sum(p.numel() for p in self.model.parameters()))
        print("\t *Trainable Params* = ",sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        
        
    def train(self, user, film, target, epoch = 1):
        '''
        

        Parameters
        ----------
        user : TYPE
            DESCRIPTION.
        film : TYPE
            DESCRIPTION.
        epoch : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        '''
        
        self.user = DataLoader(user, batch_size=self.config.batch_size, shuffle = True, num_workers=4)
        
        start = time.time()
        
        for i, user_batch in enumerate(self.user):
            recovered_matrix = self.model(user_batch, film, rating_range = 5)
            
            mse = self.lossfn(recovered_matrix, target)
            self.optimizer.zero_grad()
            mse.backward()
            self.loss_log.append(mse.detach())
            
            
            r = torch.round(recovered_matrix.detach().cpu())
            real_mse = nn.MSELoss(r, target)
            print(f'\n\t Epoch : {epoch + 1}')
            print(f'\n\t RMSE : {round(math.sqrt(real_mse),4)}')
            print('\t Training time current epoch: {round((time.time()-start),2)}')
            
            if (epoch+1) % self.config.checkpoint_freq == 0:
                
                print(f'\n\t Epoch : {epoch + 1}')
                print(f'\n\t RMSE : {round(math.sqrt(real_mse),4)}')
                print('\t Training time current epoch: {round((time.time()-start),2)}')
            
                
                self.save_checkpoint(self.model, self.optimizer, epoch+1, self.file+str(epoch+1)+'.pt')
        
        
        
    def predict(self, user, film, label):
        
        
        start = time.time()
        self.model.eval()
        r = self.model(user, film)
        
        rmse = nn.MSELoss(r, label)
        print(f'\n\t RMSE : {round(math.sqrt(rmse),4)}')
        print('\t Evaluation time: {round((time.time()-start),2)}')
        
        
        
        
        
    
    def preprocesser(self, dataset):
        
        rating_matrix = np.zeros((dataset.user_num, dataset.item_num))
        user_matrix = np.zeros((dataset.user_num, self.config.user_feature_size))
        item_matrix = np.zeros((dataset.item_num, self.config.item_feature_size))
        
        for i, data in enumerate(dataset):
            
            rating_matrix[data['user_id']-1,data['item_id']-1] = data['rating']
            
        for i, user in enumerate(dataset.users):
            user_matrix[i,:] = np.concatenate([np.atleast_1d(item).ravel() for item in user])
            
        for i, item in enumerate(dataset.items):
            item_matrix[i,:] = np.concatenate([np.atleast_1d(i).ravel() for i in item])

        
        return rating_matrix, user_matrix, item_matrix
         
            
        
            
    
    
    def config(self, configuration_file):
        
        with open(configuration_file, 'r') as file:
            config = yaml.safe_load(file)
            
        

        self.batch_size = config.get('batch_size')
        self.learning_rate = config.get('lr')
        self.item_num = config.get('item_num')
        self.user_num = config.get('user_num')
        self.item_feature_size = config.get('item_feature_size')
        self.user_feature_size = config.get('user_feature_size')
        self.AE_bottleneck = config.get('bottle_neck_size')



    
    def save_checkpoint(self, model, optimizer, epoch, filename):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
            }
        torch.save(checkpoint, filename)
        
        

    def load_checkpoint(self, filename, model, optimizer=None):
    
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            return model, optimizer, epoch
    

        
        
        
        
        
        
        
        