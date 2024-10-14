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


from torch.utils.data import DataLoader, Subset 
import yaml
from tqdm import tqdm
from AutoEncoders import DAE, DAE_KAN


class AE_KAN:
    
    def __init__(self, configuration_file=None):

        #torch.manual_seed(42)
        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.config(configuration_file)
        #self.dataset = DataLoader(data) 
        user_feature_size = self.user_feature_size
        item_feature_size = self.item_feature_size
        num_item = self.item_num
        batch = self.batch_size
        
        if self.AEmodel == 'DAE':
        
            self.model = DAE(num_item, self.AE_bottleneck,
                noise = self.noise_level).to(self.device)
        
        elif self.AEmodel == 'DAE_KAN':
            
            self.model = DAE_KAN(num_item, self.AE_bottleneck,
                noise = self.noise_level).to(self.device)
            
        #self.model = nn.DataParallel(self.model,device_ids=[0,1,2,3]) #when there are multiple gpus to use....
        self.target = None
        self.user = None
        self.film = None
        self.mask = None
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.lossfn = nn.MSELoss()
        self.loss_log = []
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10) #learning rate scheduler
        self.result = []
        self.last_output = None
     
        
        
        print("\n \t ----------- Model Loaded ------------")
        print("\t *Total Params* = ",sum(p.numel() for p in self.model.parameters()))
        print("\t *Trainable Params* = ",sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        
        
    def process_data(self, data):
    
        self.target = data[0].clone().detach().to(self.device).float()
        self.mask = data[1].clone().detach().to(self.device).float()
       
        
        
    def mse_loss(self, predict, target, mask):

        ''' Calculate MSE only for non-empty entries '''
        
        mse = torch.sum(mask * (predict - target) ** 2) / torch.sum(mask)
        return mse
        
    
    def mae_loss(self, predict, target, mask):

        ''' Calculate MSE only for non-empty entries '''
        
        mae = torch.sum(mask * torch.abs(predict - target)) / torch.sum(mask)
        return mae


    def accuracy(self, predict, target, mask):
        '''calculate the precision of the rating'''
        
        EQ = torch.eq(torch.round(predict), target)* mask
        
        return torch.sum(EQ), torch.sum(mask)
        


        
        
    def train(self, dataset, test_dataset ,epoch = 10):
        '''
        

        Parameters
        ----------
        user : torch.tensor
            the tensor storing user preference
        film : torch.tensor
            the tensor storing film representation
        epoch : int, optional
            number of training epoch. The default is 1.

        Returns
        -------
        None.

        '''
       
     

        for i in range(epoch):

            # Set the model to training mode
            self.model.train()
            epoch_loss = 0
            start = time.time()
            
            acc = 0
            tot = 0
            

            with tqdm(dataset, unit="batch") as tepoch:
                for count, data in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {i + 1}")

                    self.process_data(data)
                   
      
                        
                    if self.AEmodel == 'DAE':
                        recovered_matrix, _, _ = self.model(self.target.unsqueeze(1))
                        
                        loss = self.mse_loss(recovered_matrix.squeeze(1), self.target, self.mask)
                        
                    elif self.AEmodel == "DAE_KAN":
                        recovered_matrix, _, _ = self.model(self.target.unsqueeze(1))
                        
                        loss = self.mse_loss(recovered_matrix.squeeze(1), self.target, self.mask)
                    else:
                        raise ValueError('autoencoder = DAE, or DAE_KAN')
                    
                    matrix = torch.clamp(recovered_matrix, 1, 5)
                    
                    self.result.append(matrix)
                    
                        
                    # L2 Regularization
                    l2_loss = torch.sum(torch.tensor(0.0, requires_grad=True)).to(self.device)
                    for param in self.model.parameters():
                        l2_loss = l2_loss + torch.sum(param ** 2).to(self.device)
           
                        
                    loss_tot = loss + self.lambda_l2 * l2_loss
                    
                    a, t = self.accuracy(matrix.detach().squeeze(1), self.target, self.mask)
                    acc += int(a)
                    tot += int(t)

                    epoch_loss += loss.detach().item()
                    self.optimizer.zero_grad()
               
                    
                    loss_tot.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scheduler.step(loss)
                    self.optimizer.step()
                    
                    

                    tepoch.set_postfix(loss=loss.item()/(count+1))


            
            avg_epoch_loss = math.sqrt(epoch_loss/(1+count))
            self.loss_log.append(avg_epoch_loss)
            
            end = time.time()

            print(f'\n\t Epoch : {i + 1}, Average Training RMSE = {round(avg_epoch_loss, 4)}, "Accuracy" = {round(acc/tot, 4)}')
         
            print(f'\t Training time for current epoch: {round(end - start, 2)} seconds')
            
            #------eval-------------------------------------
            self.predict(test_dataset, recovered_matrix)

            if (i + 1) % self.save_freq == 0:
                print(f'checkpoint saved at epoch {i+1}.')
                self.save_checkpoint( self.optimizer, i + 1, f'{self.file}{i + 1}_{self.run_id}.pt')
        
        
        
    def predict(self, test_dataset, prediction):
        
     

        # Set the model to evaluation mode
        self.model.eval()
        test_loss = 0
        total_entries = 0
        mae = 0
        
        acc = 0
        tot = 0
        loss_after_round = 0
        with torch.no_grad():
       
            for i, data in enumerate(test_dataset):
                
                #user = data[0].clone().detach().to(self.device).float()
                target = data[0].clone().detach().to(self.device).float()
                mask = data[1].clone().detach().to(self.device).float()
                #film = film.clone().detach().to(self.device).float()
                
                total_entries = total_entries + torch.sum(mask)
                # Check if there are NaNs in inputs

                if torch.isnan(target).any():
                    raise ValueError("NaNs detected in target data.")
                if torch.isnan(mask).any():
                    raise ValueError("NaNs detected in mask data.")


            
            
           
                #r = torch.round(recovered_matrix.squeeze(1))
                r = prediction.squeeze(1)
                loss = self.mse_loss(r, self.target, self.mask)*torch.sum(mask)
                m = self.mae_loss(r, self.target, self.mask)*torch.sum(mask)
                loss_r = self.mse_loss(torch.round(r), self.target, self.mask)*torch.sum(mask)
                a, t = self.accuracy(r.detach(), self.target, self.mask)
                acc += int(a)
                tot += int(t)
                test_loss =  test_loss + loss
                mae = mae + m
                loss_after_round = loss_after_round + loss_r

            self.last_output = [torch.round(r), self.target, self.mask]
              
      
   
        
        rmse = math.sqrt(test_loss/total_entries)

        rrmse = math.sqrt(loss_after_round/total_entries) #round the ratings
        print(f'\t RMSE on testing set : {round(rmse,4)}, RMSE (rounded): {round(rrmse, 4)} , MAE: {round(mae.item()/total_entries.item(),4)} , Accuracy: {round(acc/tot, 4)}')
        
        
        
        
        
    
    def preprocessor(self, dataset, fullset, train = True):
        '''The preprocessor. put the dataset in to desire input form'''
        
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
           
            user_matrix[i,:45] = np.concatenate([np.atleast_1d(item).ravel() for item in user])[1:]
            
    
        for i, item in enumerate(fullset.items.values):
            vec = np.zeros((20, ),dtype='float64')
            for idx, j in enumerate(item[2]):
                vec += j*bases[idx]
            item[2] = vec
        
            item_matrix[i,:25] = np.concatenate([np.atleast_1d(i).ravel() for i in item])[1:]

        user_matrix[:,45:] = user_genre
        item_matrix[:,25] = movie_avg
        
        return rating_matrix, user_matrix, item_matrix, mask
         
            
        
    
    def config(self, configuration_file=None):
        '''The configuration class'''
        
        if not configuration_file:
            #default for 100K Dataset
            
            self.batch_size = 128
            self.learning_rate = 1e-3
            self.item_num = 1682
            self.user_num = 943
            self.item_feature_size = 26
            self.user_feature_size = 59
            self.AEmodel = 'DAE'
            self.AE_bottleneck = 512
            self.file = r'/content/drive/MyDrive/RecSys/model_checkpoints/VAE_' #for google colab
            self.save_freq = 90
            self.lambda_l2 = 0.0001
            self.test_ratio = 0.2
            self.dataset_size = 100000
            self.noise_level = 0.01
            self.embedding_dim = 24
            self.run_id = ''
            
            
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
            self.file = config.get('checkpoint_prefix')
            self.save_freq = config.get('checkpoint_save_every')
            self.lambda_l2 = config.get('lambda')
            self.test_ratio = config.get('test_split')
            self.dataset_size = config.get('data_size')
            self.noise_level = config.get('noise_level')
            self.embedding_dim = config.get('embedding_dim')
            self.run_id = config.get('run_id')



    
    def save_checkpoint(self, optimizer, epoch, filename):
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'device': self.device
        }
        torch.save(checkpoint, filename)
        
        

    def load_checkpoint(self, checkpoint_file, optimizer=None, new_device = None):
        
        filename = checkpoint_file
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
        
        return self.model, optimizer, epoch

  
    

        
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
        
        
        