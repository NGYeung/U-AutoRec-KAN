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
import pandas as pd

from model.Matrix_models import MatrixFactorization_VAE
from torch.utils.data import DataLoader, Subset 
import yaml
from tqdm import tqdm



class AEReco:
    
    def __init__(self, configuration_file=None):
        
        

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.config()
        #self.dataset = DataLoader(data) 
        user_feature_size = self.user_feature_size
        item_feature_size = self.item_feature_size
        num_item = self.item_num
        batch = self.batch_size
        
        
        self.model = MatrixFactorization_VAE(
            user_feature_size, item_feature_size, num_item, user_batch_size = batch, 
            AE_hidden_dimension = self.AE_bottleneck, autoencoder = self.AEmodel
                                             ).to(self.device)
        #self.model = nn.DataParallel(self.model,device_ids=[0,1,2,3])
        self.target = None
        self.user = None
        self.film = None
        self.mask = None
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)#add learning rate and other options
        self.lossfn = nn.MSELoss()
        self.loss_log = []
        
        
        print("\n \t ----------- Model Loaded ------------")
        print("\t *Total Params* = ",sum(p.numel() for p in self.model.parameters()))
        print("\t *Trainable Params* = ",sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        
        
    def process_data(self, data, film):
        # Convert numpy arrays to torch tensors and move to device
        self.user = data[0].clone().detach().to(self.device).float()
        self.target = data[1].clone().detach().to(self.device).float()
        self.mask = data[2].clone().detach().to(self.device).float()
        self.film = film.clone().detach().to(self.device).float()
        #print(self.user.shape, self.target.shape, self.mask.shape, self.film.shape)

        
        
    def train(self, dataset, film, epoch = 10):
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
        # TODO: randomly shuffle the user; so the target is also different.
        
        mse_loss_func = nn.MSELoss()

        for i in range(epoch):
            epoch_loss = 0
            start = time.time()

            with tqdm(dataset, unit="batch") as tepoch:
                for count, data in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {i + 1}")

                    self.process_data(data, film)
                    if torch.isnan(self.user).any():
                        print(f"NaNs in user at epoch {i}, batch {count}")
                    if torch.isnan(self.film).any():
                        print(f"NaNs in film at epoch {i}, batch {count}")

                    if self.AEmodel == 'VAE':
                        recovered_matrix, mean, log_var = self.model(self.user, self.film, rating_range=5)
                        KL_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                        loss = self.lossfn(recovered_matrix * self.mask.view(-1), self.target.view(-1)) + KL_div

                        if torch.isnan(loss).any():
                            print("NaNs found in total_loss")
                            break
                    else:
                        recovered_matrix, _, _ = self.model(self.user, self.film, rating_range=5)
                        loss = self.lossfn(recovered_matrix * self.mask.view(-1), self.target.view(-1))

                    entry = torch.count_nonzero(self.mask)
                    total = self.mask.shape[0]*self.mask.shape[1]
                    epoch_loss += (loss.detach().item())**2/entry*total
                    self.optimizer.zero_grad()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    loss.backward()
                    self.optimizer.step()
                    self.loss_log.append(epoch_loss / (count + 1))

                    tepoch.set_postfix(loss=loss.item())

            r = torch.round(recovered_matrix.detach() * self.mask.view(-1))
            real_mse = mse_loss_func(r, self.target.view(-1))

            avg_epoch_loss = math.sqrt(epoch_loss.item())
            rmse = math.sqrt(real_mse.item())/entry*total
            end = time.time()

            print(f'\n\t Epoch : {i + 1}, RMSE = {round(avg_epoch_loss, 4)}')
            #print(f'\t RMSE : {round(rmse.item(), 4)}')
            print(f'\t Training time for current epoch: {round(end - start, 2)} seconds')

            if (i + 1) % self.save_freq == 0:
                print(f'checkpoint saved at epoch {i+1}.')
                self.save_checkpoint( self.optimizer, i + 1, f'{self.file}{i + 1}.pt')
        
        
        
    def predict(self, user, film, mask, label):
        
        
        start = time.time()
        self.model.eval()
        r = self.model(user, film)
        
        rmse = nn.MSELoss(r * mask.view(-1) , label.view(-1))
        print(f'\n\t RMSE : {round(math.sqrt(rmse),4)}')
        print('\t Evaluation time: {round((time.time()-start),2)}')
        
        
        
        
        
    
    def preprocessor(self, dataset, fullset, train = True):
        
        rating_matrix = np.zeros((fullset.user_num, fullset.item_num))
        mask = np.zeros((fullset.user_num, fullset.item_num))
        user_matrix = np.zeros((fullset.user_num, self.user_feature_size))
        item_matrix = np.zeros((fullset.item_num, self.item_feature_size))
        user_genre_ct = np.zeros((fullset.user_num, 19))
        user_genre_rt = np.zeros((fullset.user_num, 19))
        item_genre = np.zeros((fullset.user_num, 20))
        user_genre = np.zeros((fullset.user_num, 20))
        movie_avg = np.zeros((fullset.item_num,))
        movie_ct = np.zeros((fullset.item_num,))
        
        
        bases = fourier_bases(19, 20)

        for i, data in enumerate(dataset):
            
            rating_matrix[data['user_id']-1,data['movie_id']-1] = data['rating']
            mask[data['user_id']-1,data['movie_id']-1]  = 1
            if train:
                user_genre_ct[data['user_id']-1,:] += np.array(data['genre'])
                user_genre_rt[data['user_id']-1,:] += data['rating']*np.array(data['genre'])
                movie_ct[data['movie_id']-1] += 1
                movie_avg[data['movie_id']-1] += data['rating']
            
                
            
            
        
        user_genre_rt = user_genre_rt / (user_genre_ct+1e-12)
        movie_avg /= (movie_ct + 1e-12)
        
        for idd, i in enumerate(user_genre_rt):
            vec = np.zeros((20,))
            for idx, j in enumerate(i):
                vec += j*bases[idx,:]
            user_genre[idd,:] += vec
            
            
        for i, user in enumerate(fullset.users.values):
            #print(np.concatenate([np.atleast_1d(item).ravel() for item in user]))
            user_matrix[i,:39] = np.concatenate([np.atleast_1d(item).ravel() for item in user])[1:]
            
    
        for i, item in enumerate(fullset.items.values):
            vec = np.zeros((20, ),dtype='float64')
            for idx, j in enumerate(item[2]):
                vec += j*bases[idx]
            item[2] = vec
        
            item_matrix[i,:25] = np.concatenate([np.atleast_1d(i).ravel() for i in item])[1:]

        user_matrix[:,39:] = user_genre
        item_matrix[:,25] = movie_avg
        
        return rating_matrix, user_matrix, item_matrix, mask
         
            
        
    
    
    
    
    def config(self, configuration_file=None):
        
        if not configuration_file:
            #default for 100K Dataset
            
            self.batch_size = 64
            self.learning_rate = 1e-5
            self.item_num = 1682
            self.user_num = 943
            self.item_feature_size = 26
            self.user_feature_size = 59
            self.AEmodel = 'DAE'
            self.AE_bottleneck = 512
            self.file = r'/content/drive/MyDrive/RecSys/model_checkpoints/VAE_' #for google colab
            self.save_freq = 30
        
        
        else:
            with open(configuration_file, 'r') as file:
                config = yaml.safe_load(file)
            
        

            self.batch_size = config.get('batch_size')
            self.learning_rate = config.get('lr')
            self.item_num = config.get('item_num')
            self.user_num = config.get('user_num')
            self.item_feature_size = config.get('item_feature_size')
            self.user_feature_size = config.get('user_feature_size')
            self.AEmodel = config.get('AutoEncoder')
            self.AE_bottleneck = config.get('bottle_neck_size')



    
    def save_checkpoint(self, optimizer, epoch, filename):
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'device': self.device
        }
        torch.save(checkpoint, filename)
        
        

    def load_checkpoint(self, checkpoint_num, optimizer=None, new_device = None):
        
        filename = self.file+str(checkpoint_num)+'.pt'
        if new_device:
            self.device = new_device
    
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        
        checkpoint_device = checkpoint.get('device', 'cpu')
        print(f"Checkpoint saved on device: {checkpoint_device}")
        print(f"Current device: {self.device}")
        
        return model, optimizer, epoch

  
    

        
#--------------------------------Helper Function-----------------------------------     
        
        
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


def custom_collate(batch, batch_size, dataset):
    if len(batch) < batch_size:
      
        additional_samples_needed = batch_size - len(batch)
        indices = np.random.choice(len(dataset), additional_samples_needed)
        additional_data = [dataset[idx] for idx in indices]
        batch.extend(additional_data)
    return default_collate(batch)
        
        
        