# User-based collaborative filtering recommender system with Autoencoder

For reinformance-based recommendation system, please refer to https://github.com/NGYeung/RL-playground/tree/main

This repository provides an implementation of a user-based CF rating prediction algorithm using and KAN based autoencoder.

*Repository Structure:*

	Training Notebook - Recommender.ipynb
	Model files:
		Recommender.py 
		model\AutoEncoders.py
		model\Matrix_Models.py
	Dataset:
		MovieLens.py
		MovieLens4Colab.py
	Raw Data:
		ml-100k\.
		ml-1M\.

## An implementation of U-Autorec with with KAN Layers.

*Dataset: [MovieLens100K](https://grouplens.org/datasets/movielens/100k/) and [MovieLens1M](https://grouplens.org/datasets/movielens/1m/)*

*The Implementation of KAN: [FastKan](https://github.com/ZiyaoLi/fast-kan)* 

*Train-test split: 9:1*

**100K Best Result (Updating) RMSE: 0.9811**

**1M Best Result (Updating) RMSE: 0.9535**

**For reference: 1M Dataset same parameters with nn.Linear: RMSE 0.9643**

