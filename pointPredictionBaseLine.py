# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:38:28 2019

@author: Jeremy
"""

from sklearn import datasets
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

import proPack.comparator as cmp
import proPack.dataManager as dm
from proPack.regressor.dummyRegressor import DummyRegressor
from proPack.composite.compositeEstimator import CompositeProbEstimator, ClassicalBaseline, ConstantUninformedBaseline

#parameters
CV = 5
RDG_GAMMA = 3.0517578125e-05
RDG_ALPHA = 0.00048828125
RDG_KERNEL = 'rbf' #'rbf', 'linear'

lossFunctions = ['logloss', 'clipped_logloss', 'integrated_loss']
lossFunction = lossFunctions[1]
distribution  = 'normal' #['normal', 'laplace']

#comparaison
data = dm.dataManager(dataSet = datasets.load_boston(), normalize_data =True)
comp = cmp.ModelComparator(data, lossFunction)

comp.addModel('classicalBaseLine', ClassicalBaseline(distribution))
comp.addModel('uninformedBaseLine', ConstantUninformedBaseline(42, distribution))

comp.addModel('linearRegression', CompositeProbEstimator(LinearRegression(), DummyRegressor(constant = 42), distribution))
comp.addModel('randomForest.Self.Trained', CompositeProbEstimator(RandomForestRegressor(max_depth=20, n_estimators=80, random_state=2), DummyRegressor('std'), distribution))
comp.addModel('kernelRidge.Self.Trained', CompositeProbEstimator(KernelRidge(alpha = RDG_ALPHA, kernel = RDG_KERNEL, gamma = RDG_GAMMA), DummyRegressor('std'), distribution))

rdm_param_grid = {'max_depth': (10, 20, None), 'n_estimators': (5, 20, 30)}
rdm = GridSearchCV(estimator= RandomForestRegressor(random_state=100), param_grid = rdm_param_grid, cv= CV)
comp.addModel('randomForest', CompositeProbEstimator(rdm, DummyRegressor('std'), distribution))

kr_param_grid = { 'alpha': (10**(-10), 0.1, 0.01) }
kr1 = GridSearchCV(estimator=KernelRidge(kernel = RDG_KERNEL, gamma = RDG_GAMMA), param_grid= kr_param_grid, cv= CV)
comp.addModel('kernelRidge', CompositeProbEstimator(kr1, DummyRegressor('std'), distribution))


# compare
comp.fit()
comp.evaluate()
comp.crossValidation(n_splits = CV)

print('')
print(comp)