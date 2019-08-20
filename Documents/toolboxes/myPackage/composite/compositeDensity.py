# -*- codinsuper()g: utf-8 -*-
import numpy as np
    
class CompositeDensity :
    
     def __init__(self, loc, scale):
        self.loc = loc 
        self.scale = scale
        
class NormalCompositeDensity(CompositeDensity) :
    
     def __init__(self, loc, scale):
         super().__init__(loc, scale)
         
     def __call__(self, y):
        return np.exp(-(y - self.loc)**2/(2*self.scale))/np.sqrt(2*np.pi*self.scale)
    
     def squared_norm(self):
        return 1/(2**self.scale*np.sqrt(np.pi))
    
class LaplaceCompositeDensity(CompositeDensity) :
    
     def __init__(self, loc, scale):
         super().__init__(loc, scale)
    
     def __call__(self, y):
        return  np.exp(-np.abs(y -self. loc)/self.scale)/(2*self.scale)


     def squared_norm(self):
        return 1/(4*self.scale)
  
    
if __name__ == "__main__" :
    print("class composite density compile test ...")
    import myPackage.lossFunctions as lf
    
    p = NormalCompositeDensity(0.0, -0.5)
    print(not isinstance(p, CompositeDensity))
    
    y = 0.2
    logloss = lf.compositeDensityFunctors()['logloss']
    clogloss = lf.compositeDensityFunctors()['clipped_logloss']
    
    print('p:', p(y))
    print('logloss:', logloss(p,y))
    print('clogloss:', clogloss(p,y))
