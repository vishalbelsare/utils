# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:39:34 2019

@author: Jeremy
"""

import numpy as np
from sklearn import datasets
from sklearn.linear_model.base import BaseEstimator, LinearRegression, RegressorMixin, _rescale_data
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import safe_sparse_dot

import scipy.sparse as sp
from scipy.optimize import minimize




class RobustRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, fit_intercept = True, normalize=False, copy_X=True, 
                 optim_method = 'Nelder-Mead', optim_args = {}):
        
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.optim_method = optim_method
        self.optim_args = optim_args
 
    
    def fit(self, X, y, sample_weight=None):
        
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         y_numeric=True, multi_output=True)
        
        if self.copy_X :
            if sp.issparse(X):
                X = X.copy()
            else:
                X = X.copy(order='K')

        if self.normalize == True :
            X = normalize(X)
        
        if self.fit_intercept :
            X = np.column_stack((np.ones(X.shape[0]), X)) 

        if sample_weight is not None:
            if np.atleast_1d(sample_weight).ndim > 1: 
                raise ValueError("Sample weights must be 1D array or scalar")
            
            X, y = _rescale_data(X, y, sample_weight)
        
        optim_option = self.optim_args.copy()
        optim_option['fun'] = lambda x : self._l1_loss(betas=x, X=X, y=y)
        optim_option['jac'] = lambda x : self._jac(betas=x, X=X, y=y)
        optim_option['method'] = self.optim_method
        
        if 'x0' not in optim_option :
            optim_option['x0'] = self._x0_ls_guess(X, y)
        
        self.res = minimize(**optim_option)
        
        if self.fit_intercept :
            self.coef_ = self.res.x[1:]
            self.intercept_ = self.res.x[0]
        else:
            self.coef_ = self.res.x
            self.intercept_ = 0.0

        return self
    

    def predict(self, X):
        check_is_fitted(self, "coef_")

        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        
        if self.copy_X :
            if sp.issparse(X):
                X = X.copy()
            else:
                X = X.copy(order='K')
        
        if self.normalize == True :
            X = normalize(X)

        return safe_sparse_dot(X, self.coef_.T,
                               dense_output=True) + self.intercept_
 
    
    @staticmethod
    def _x0_ls_guess(X, y) :
        lse = LinearRegression(fit_intercept = True, normalize= True)
        lse.fit(X,y)
        return lse.coef_

    
    @staticmethod
    def _l1_loss(betas, X, y):
        return sum(abs(y - X.dot(betas)))
    
    @staticmethod
    def _jac(betas, X, y):
        mat = np.sign((y - X.dot(betas)))
        return (mat.T).dot(X)
        

if __name__ == "__main__" :
    print("class robustReg compile test ...")
    
    reg = RobustRegressor(fit_intercept = False)
   
    data = datasets.load_boston()
    X = data.data 
    y = data.target
    
    Xh = normalize(X)
    reg.fit(Xh,y)
    pred = reg.predict(X)