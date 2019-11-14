# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 16:53:54 2019

@author: Jeremy
"""
import numpy as np

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize

import myPackage.comparator as cmp
import myPackage.dataManager as dm
from myPackage.composite.compositeEstimator import CompositeProbEstimator


data = dm.dataManager(dataSet = datasets.load_boston(), normalize_data =True)

#1. Linear composite
compLR = cmp.ModelComparator(data, 'logloss')
estimator =  CompositeProbEstimator(LinearRegression(), LinearRegression(), 'normal' )
compLR.addModel('p=LR_s=LR', estimator)
compLR.fit()
compLR.evaluate()
print(compLR )

#re-implementation
y = data.Y_train.copy()
X = data.X_train.copy()
X = normalize(X)
            
meanEstimator = LinearRegression()
varianceEstimator = LinearRegression()

meanEstimator.fit(X,y)
sqr_prediction = np.square(meanEstimator.predict(X) - y)

varianceEstimator.fit(X,sqr_prediction)
variancePrediction = varianceEstimator.predict(X)