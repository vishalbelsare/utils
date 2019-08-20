# -*- coding: utf-8 -*-


import numpy as np
import myPackage.composite.compositeDensity as cd


     
class MeanSquareError() :

     def __call__(self, f, y) :
        return np.square(y - f)


class ProbLossFunction :
    
     def __init__(self):
        self.type = 'probabilistic'
        
     def check_density(self,f):
        if not isinstance(f, cd.CompositeDensity):
            raise ValueError("prediction entry is not a denstiy functor")
        pass
        
        
class LogLoss(ProbLossFunction) :

     def __call__(self, f, y) :
        self.check_density(f)
        return -np.log(f(y))


class LogLossClipped(ProbLossFunction) :
    
    def __init__(self, cap = np.exp(-23)):
        self.cap = cap

    def __call__(self, f, y) :
        self.check_density(f)
        return np.clip(a = -np.log(f(y)), a_max = -np.log(self.cap), a_min = None)
   
    
class IntegratedSquaredLoss(ProbLossFunction) :

    def __call__(self, f, y) :
        self.check_density(f)
        return - 2 * f(y) + f.squared_norm()



class compositeDensityFunctors():
    
    __functorsDic = {'mean_square': MeanSquareError(), 
                     'logloss' : LogLoss(),
                     'clipped_logloss' : LogLossClipped(),
                     'integrated_loss': IntegratedSquaredLoss()
                     }
        
    def __init__(self):
        pass
    
    def __getitem__(self, key):
        if key not in self.__functorsDic.keys():
            raise ValueError("Unknown loss type: %s, expected one of %s."
                             % (key, self.__functorsDic.keys()))
        return self.__functorsDic[key]
        
        
