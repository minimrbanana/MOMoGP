"""
Created on June 08, 2021
@author: Zhongjie Yu
@author: Mingye Zhu
"""
#import gc
#import itertools
import numpy as np
import pandas as pd
#from gpytorch.likelihoods import GaussianLikelihood
from MOMoGPstructure import query, build_bins
from MOMoGP import structure#, ExactGPModel
import random
#from scipy.special import logsumexp
import torch
#from torch import optim
#from torch.optim import *
#import gpytorch
#from gpytorch.mlls import *
#from scipy.io import arff
#from sklearn.decomposition import PCA
#import dill
from utils import calc_rmse, calc_mae, calc_nlpd

random.seed(23)
np.random.seed(23)
torch.manual_seed(23)
torch.cuda.manual_seed(23)


# load data
x_train = pd.read_csv('./data/Parkinsons/x_train.csv')
x_test = pd.read_csv('./data/Parkinsons/x_test.csv')
y_train = pd.read_csv('./data/Parkinsons/y_train.csv')
y_test = pd.read_csv('./data/Parkinsons/y_test.csv')
print('training data shape:', x_train.shape)
print('test data shape:', x_test.shape)

# normalise data
mu_x,std_x = x_train.mean().to_numpy(), x_train.std().to_numpy()
mu_y,std_y = y_train.mean().to_numpy(), y_train.std().to_numpy()
x_train = (x_train - mu_x)/std_x
x_test = (x_test - mu_x)/std_x
y_train = (y_train - mu_y)/std_y
y_test = (y_test - mu_y)/std_y
x_train = x_train.iloc[:,:].values
x_test = x_test.iloc[:,:].values
y_train = y_train.iloc[:,:].values
y_test = y_test.iloc[:,:].values
y_d = y_train.shape[1]
d_input = x_train.shape[1]

# hyperparameter settings
lr = 0.1
rerun = 1
epoch =200
RMSEE=[]
MAEE=[]
NLPDD=[]
scores = np.zeros((rerun,3))

for k in range(rerun):
    # built the root structure
    opts = {
        'min_samples': 0,
        'X': x_train,
        'Y': y_train,
        'qd': 1,
        'max_depth': 100,
        'max_samples': 550,
        'log': True,
        'jump': True,
        'reduce_branching': True
    }
    root_region, gps_ = build_bins(**opts)
    root, gps = structure(root_region,scope = [i for i in range(y_d)], gp_types=['matern1.5_ard'])

    # train GP experts with their own hyperparameters
    outer_LMM = np.zeros((len(gps),epoch))
    for i, gp in enumerate(gps):
        idx = query(x_train, gp.mins, gp.maxs)
        gp.x = x_train[idx]
        y_scope = y_train[:,gp.scope]
        gp.y = y_scope[idx]
        print(f"Training GP {i + 1}/{len(gps)} ({len(idx)})")
        outer_LMM[i,:]= gp.init(cuda=True,lr = lr, steps=epoch,iter = i)
    root.update()

    # on test data
    mu, cov= root.forward(x_test[:,:], smudge=0,y_d = y_d)

    # evaluate RMSE, MAE and NLPD
    RMSE = calc_rmse(mu, y_test)
    MAE = calc_mae(mu, y_test)
    NLPD = calc_nlpd(mu, cov, y_test)
    print(RMSE, MAE, NLPD)

    RMSEE.append(RMSE)
    MAEE.append(MAE)
    NLPDD.append(NLPD)
    scores[k,0] = RMSE
    scores[k,1] = MAE
    scores[k,2] = NLPD

# print ecaluation
# uncomment if you want to save the evaluation
#np.savetxt('MOMoGP_scores_parkinsons.csv', scores, delimiter=',')
print(f"MOMoGP  RMSE: {RMSEE}")
print(f"MOMoGP  MAE: {MAEE}")
print(f"MOMoGP  NLPD: {NLPDD}")
print(f"MOMoGP  RMSE mean: {np.mean(np.array(RMSEE))} std:{np.std(np.array(RMSEE))}")
print(f"MOMoGP  MAE mean: {np.mean(np.array(MAEE))} std:{np.std(np.array(MAEE))}")
print(f"MOMoGP  NLPD mean: {np.mean(np.array(NLPDD))} std:{np.std(np.array(NLPDD))}")
