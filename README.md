# User-based collaborative filtering recommender system with Denoising Autoencoder with KAN Layers

For reinformance-based recommendation system, please refer to https://github.com/NGYeung/RL-playground/tree/main

This repository provides an implementation of a user-based CF rating prediction algorithm using and KAN based autoencoder.

*Repository Structure:*

	Training Notebook - Recommender.ipynb
	Model files:
		Recommender.py 
		model\AutoEncoders.py
	Dataset:
		MovieLens.py
		MovieLens4Colab.py
	Raw Data:
		ml-100k\.
		ml-1M\.


### An Illustration of the model


### Datasets and Results

*Dataset: [MovieLens100K](https://grouplens.org/datasets/movielens/100k/) and [MovieLens1M](https://grouplens.org/datasets/movielens/1m/)*

*The Implementation of KAN: [FastKan](https://github.com/ZiyaoLi/fast-kan)* 

*Train-test split: 9:1*

**100K Best Result RMSE: TBD**

**MovieLens 1M Result RMSE: 0.7921 (Average over 5 cross-validations) !!!!!! **



