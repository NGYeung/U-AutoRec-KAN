# User-based collaborative filtering recommender system with Autoencoder

For reinformance-based recommendation system, please refer to https://github.com/NGYeung/RL-playground/tree/main

This repository provides an implementation of a user-based CF rating prediction algorithm using linear autoencoder and KAN based autoencoder.

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
		

## An implementation of U-Autorec with Fourier encoding of user/item info.

*Dataset: MovieLens100K*

*Train-test split: 8:2*

**Best Result (Updating) RMSE: 1.0081**

## U-Autorec with KAN Layers and Fourier encoding or user/item info.

*Dataset: MovieLens100K*

*Train-test split: 8:2*

**Best Result (Updating) RMSE: 0.9811**
