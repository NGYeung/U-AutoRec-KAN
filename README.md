# User-based collaborative filtering recommender system with Denoising Autoencoder with KAN Layers

For reinformance-based recommendation system, please refer to https://github.com/NGYeung/RL-playground/tree/main

This repository provides an implementation of a user-based CF rating prediction algorithm using and KAN based autoencoder.

*Repository Structure:*

	Training Notebook - Recommender.ipynb
 
	Model files:
		Recommender.py 
		model\AutoEncoders.py
  		model\pre_trained.pt
	Dataset:
		MovieLens4Colab.py 
  		(Including dataset construction and pre-processing method. Need to replace the link to source files for your own use.)
	Configuration:
 		MovieLens100K.yaml
   		MovieLens1M.yaml
		


### An Illustration of the model


### Datasets and Results

*Dataset: [MovieLens100K](https://grouplens.org/datasets/movielens/100k/) and [MovieLens1M](https://grouplens.org/datasets/movielens/1m/)*

*The Implementation of KAN: [FastKan](https://github.com/ZiyaoLi/fast-kan)* 

*Train-test split: 9:1*

**100K Best Result RMSE: 0.54**

**MovieLens 1M Result RMSE: 0.7921 (Average over 5 cross-validations **



