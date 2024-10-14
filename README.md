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


## Collaborative Filtering

Collaborative Filtering is a subset of recommender systems that suggests items to users based on the similarities among the users and items. In other words, it predicts a user's interest in an item given previous user-item interactions. It's widely applied in e-commerce and social media recommendations.

### AutoRec with MLP

[AutoRec](https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf) is proposed by S Sedhain et al. in 2015. It directly takes the documented ratings as the input and uses autoencoders' great ability at learning the latent representations to predict users' preference (ratings) of uncountered items. The problem is stated as follows:

Suppose we have $`m`$ users and $`n`$ items in the dataset, the dataset $`D`$ provides us with a partial observation of the user-item interaction matrix $`R`$, the i-th row of which is the rating from the i-th user for all items. Thus, we can conveniently denote user preferences as $`u_1, \cdots, u_m \in \mathbb{R}^n`$. 

The autoencoder receives known ratings as the input, encode them as a low-dimensional representation, and recovered the rating through a decoder structure. Let's denote the reconstruction for rating $`r`$ as $`\tilde r = h(r)`$, the algorithm is solving the following optimization problem. 

$$\min \limits_{r\in D} \| r - h(r;\theta) \|_2^2$$

If the model sticks to linear layers, the $h'(r)$ can be written as
$$h(r;\theta) = f(W+g(Vr+\mu)+b)$$
where $`\theta = \{W,V, \mu, b\}`$ and $`f`$ and $`g`$ are activition functions.

---

### An Illustration of the model

![image](model_illustration.jpg)

---

### Datasets and Results

_Dataset 1: [MovieLens1M](https://grouplens.org/datasets/movielens/1m/)_ These files contain 1,000,209 anonymous ratings of approximately 3,900 movies 
made by 6,040 MovieLens users who joined MovieLens in 2000.


_Train-test split: 9:1, 10-fold cross-validatoin for ML-1M, and the RMSE is calculated as_ $`\sqrt{0.1\sum_i RMSE_i}`$
