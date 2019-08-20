
import numpy as np

from sklearn.linear_model.base import BaseEstimator 
from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import normalize

import scipy.sparse as sp

from myPackage.composite.compositeDensity import NormalCompositeDensity, LaplaceCompositeDensity
from myPackage.regressor.dummyRegressor import DummyRegressor


class CompositeProbEstimator(BaseEstimator) :
    
    def __init__(self, meanEstimator, varianceEstimator, distribution = 'normal', 
                 minWrap = False, minWrap_constant = 0.0, normalize=False, copy_X=True):
        
        self.meanEstimator = meanEstimator 
        self.varianceEstimator = varianceEstimator
        
        self.distribution = distribution
        
        self.minWrap = minWrap
        self.minWrap_constant = minWrap_constant
        self.normalize = normalize
        self.copy_X = copy_X
        

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
            
        self.meanEstimator.fit(X,y)
        sqr_prediction = np.square(self.meanEstimator.predict(X) - y)
        
        self.varianceEstimator.fit(X,sqr_prediction)
        
        return self
    
    
    def predict(self, X):
        
        allowed_distribution = ("normal", "laplace")
        if self.distribution not in allowed_distribution:
            raise ValueError("Unknown strategy type: %s, expected one of %s."
                             % (self.distribution, allowed_distribution))

        if self.copy_X :
            if sp.issparse(X):
                X = X.copy()
            else:
                X = X.copy(order='K')
        
        if self.normalize == True :
            X = normalize(X)
        
        variancePrediction = self.varianceEstimator.predict(X)
        
        if self.minWrap == True :
             variancePrediction = np.clip(a = variancePrediction, a_max = None, a_min = self.minWrap_constant)
        
        if self.distribution == 'normal' :
            return NormalCompositeDensity(self.meanEstimator.predict(X), variancePrediction)
        
        elif self.distribution == 'laplace' :
            return LaplaceCompositeDensity(self.meanEstimator.predict(X), variancePrediction)
        
        else : return
        

class ConstantUninformedBaseline(CompositeProbEstimator) :
    
    def __init__(self, constant = 42, distribution = 'normal', normalize=False, copy_X=True):
         super().__init__(DummyRegressor(constant = constant), DummyRegressor(constant = constant), distribution, normalize, copy_X)
        

class ClassicalBaseline(CompositeProbEstimator) :
    
    def __init__(self, distribution = 'normal', normalize=False, copy_X=True):
       super().__init__(DummyRegressor('mean'), DummyRegressor('std'), distribution, normalize, copy_X)
 
    
    
if __name__ == "__main__" :
    print("class composite compile test ...")
