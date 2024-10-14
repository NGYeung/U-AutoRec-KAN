# User-based collaborative filtering recommender system with Denoising Autoencoder with KAN Layers

For reinformance-based recommendation system, please refer to https://github.com/NGYeung/RL-playground/tree/main

This repository provides an implementation of a user-based CF rating prediction algorithm using and KAN based autoencoder.

_The Implementation of KAN: [FastKan](https://github.com/ZiyaoLi/fast-kan)_



_Repository Structure:_

```plaintext
.
├── Recommender.ipynb
├── Recommender.py
├── model
│   ├── AutoEncoders.py
│   ├── pre_trained.pt
├── MovieLens4Colab.py
├── MovieLens100K.yaml
└── MovieLens1M.yaml
```

---

### An Illustration of the model

![image](model_illustration.jpg)

---

### Datasets and Results

_Dataset 1: [MovieLens1M](https://grouplens.org/datasets/movielens/1m/)_ These files contain 1,000,209 anonymous ratings of approximately 3,900 movies 
made by 6,040 MovieLens users who joined MovieLens in 2000.


_Train-test split: 9:1, 10-fold cross-validatoin for ML-1M, and the RMSE is calculated as $\sqrt{0.1\sum_i RMSE_i}$_
