# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:38:28 2019

@author: Jeremy
"""

from sklearn import datasets
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import myPackage.comparator as cmp
import myPackage.dataManager as dm
from myPackage.regressor.dummyRegressor import DummyRegressor
from myPackage.composite.compositeEstimator import CompositeProbEstimator

#parameters
CV = 5
RDG_GAMMA = 3.0517578125e-05
RDG_ALPHA = 0.00048828125

lossFunctions = ['logloss', 'clipped_logloss', 'integrated_loss']
lossFunction = lossFunctions[1]
distribution  = 'normal' #['normal', 'laplace']
winWrap_constant = 2**5

data = dm.dataManager(dataSet = datasets.load_boston(), normalize_data =True)


#1. Linear composite
compLR = cmp.ModelComparator(data, lossFunction)
compLR.addModel('p=LR_s=Std', CompositeProbEstimator(LinearRegression(), DummyRegressor('std'), distribution))
compLR.addModel('p=LR_s=LR', CompositeProbEstimator(LinearRegression(), LinearRegression(), distribution))
compLR.addModel('p=LR_s=RF', CompositeProbEstimator(LinearRegression(), RandomForestRegressor(max_depth=20, n_estimators=80, random_state=2), distribution))
compLR.addModel('p=LR_s=GP', CompositeProbEstimator(LinearRegression(), KernelRidge(alpha = RDG_ALPHA, kernel = 'rbf', gamma = RDG_GAMMA), distribution))

compLR.fit()
compLR.evaluate()
#compLR.crossValidation(n_splits = 5)

#1*. Linear composite with MinWR
compLRWM = cmp.ModelComparator(data, lossFunction)
compLRWM.addModel('p=LR_s=MinStd', CompositeProbEstimator(LinearRegression(), DummyRegressor('std'), distribution, minWrap = True, minWrap_constant = winWrap_constant))
compLRWM.addModel('p=LR_s=MinLR', CompositeProbEstimator(LinearRegression(), LinearRegression(), distribution, minWrap = True, minWrap_constant = winWrap_constant))
compLRWM.addModel('p=LR_s=MinRF', CompositeProbEstimator(LinearRegression(), RandomForestRegressor(max_depth=20, n_estimators=80, random_state=100), distribution, minWrap = True, minWrap_constant = winWrap_constant))
compLRWM.addModel('p=LR_s=MinGP', CompositeProbEstimator(LinearRegression(), KernelRidge(alpha = RDG_ALPHA, kernel = 'rbf', gamma = RDG_GAMMA), distribution, minWrap = True, minWrap_constant = winWrap_constant))

compLRWM.fit()
compLRWM.evaluate()
#compLRWM.crossValidation(n_splits = 5)

#print results
print('')
print(compLR )
print(compLRWM )
