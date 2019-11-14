# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:09:38 2019

@author: Jeremy
"""

import numpy as np

from sklearn import datasets

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.kernel_ridge import KernelRidge
from robustRegressor import RobustRegressor
from sklearn.ensemble import RandomForestRegressor

SPLIT_SEED = 5
CV = 100

DO_KERNEL = False
DO_ROBUST = False
DO_RF = False

dataset = datasets.load_boston()

X_train, X_test, Y_train, Y_test = train_test_split(
        dataset.data, 
        dataset.target, 
        test_size = 0.2, 
        random_state= SPLIT_SEED
        )

#ridge
if DO_KERNEL :
    kr = KernelRidge(alpha=1, kernel = 'rbf', gamma = 1)
    kr_param_grid = { 'alpha': np.logspace(-15, 5, 11, base = 2), 'gamma':  np.logspace(-15, 5, 11, base = 2) }
    
    print('kernelRidge parameters Grid search ...')
    krGS = GridSearchCV(estimator= kr, param_grid= kr_param_grid, cv= CV)
    krGS.fit(X_train, Y_train)
    
    print('fit done ...')
    krParam = krGS.best_params_
    print(krParam)

#robustReg
if DO_ROBUST :
    reg = RobustRegressor(fit_intercept = True, normalize = True)
    methods_grid = ['Nelder-Mead','Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']
    
    print('robustRegression parameters Grid search ...')
    regGS = GridSearchCV(estimator= reg, param_grid = {'optim_method' : methods_grid}, cv= 5)
    regGS.fit(X_train, Y_train)
    
    print('fit done ...')
    regParam = regGS.best_params_
    print(regParam)
    
#random_F
if DO_RF :    
    print('randomF parameters Grid search ...')
    rdm = RandomForestRegressor(max_depth=5, n_estimators=100, random_state=2)
    rdm_param_grid = {'max_depth': range(1,50), 'n_estimators': range(10,100,10)}

    rdmGS = GridSearchCV(estimator= rdm, param_grid = rdm_param_grid, cv= CV)
    rdmGS.fit(X_train, Y_train)
    
    print('fit done ...')
    rdmParam = rdmGS.best_params_
    print(rdmParam)
