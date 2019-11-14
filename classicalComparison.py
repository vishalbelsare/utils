# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:38:28 2019

@author: Jeremy
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:42:08 2019

@author: Jeremy
"""

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import comparator as cmp
import dataManager as dm
from robustRegressor import RobustRegressor
from dummyRegressor import DummyRegressor

#parameters
RDG_GAMMA = 3.0517578125e-05
RDG_ALPHA = 0.00048828125
NN_ACTIVATE = False
NN_BATCH_SIZE = 30
NN_EPOCHS = 500
NN_PRINT = 0
NORMALIZE = False
GPkernel=  ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")

#comparaison
data = dm.dataManager(dataSet = datasets.load_boston(), normalize_data = NORMALIZE)
comp = cmp.ModelComparator(data)
comp.addModel('dummy', DummyRegressor(strategy='mean', constant=None, quantile=None))
comp.addModel('linearOLS', LinearRegression())
comp.addModel('kernelRidge', KernelRidge(alpha = RDG_ALPHA, kernel = 'rbf', gamma = RDG_GAMMA))
comp.addModel('randomForest', RandomForestRegressor(max_depth=20, n_estimators=80, random_state=2))
comp.addModel('robustRegression', RobustRegressor(normalize = True, optim_method = 'COBYLA'))
comp.addModel('GPRegression', GaussianProcessRegressor(kernel = GPkernel, alpha=RDG_ALPHA))

#ShallowNN
if NN_ACTIVATE is True :
    snn = Sequential()
    snn.add(Dense(units=30, kernel_initializer='normal', activation='relu',input_dim=13))
    snn.add(Dense(units=10, kernel_initializer='normal', activation='relu'))
    snn.add(Dense(1, kernel_initializer='normal'))  
    
    fitArgs = {'epochs': NN_EPOCHS, 'batch_size': NN_BATCH_SIZE, 'verbose': NN_PRINT}
    snn.compile(loss='mean_squared_error', optimizer='adam')
    comp.addModel(name = 'shallowNN', model = snn, fitting_arguments = fitArgs)

# compare
comp.fit()
comp.evaluate()
comp.crossValidation()
#comp.print()
