# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:14:35 2019

@author: Jeremy
"""
import numpy as np
from sklearn.model_selection import RepeatedKFold

from myPackage.lossFunctions import compositeDensityFunctors
        


class ModelComparator:
    
    def __init__(self, dataManager, lossFunction = 'mean_square'):
        
        self.models = {}
        self.fitArgs = {}
       
        self.testResults = {}
        self.trainResults = {}
        self.CV = {}
        
        self.lossFunctor = compositeDensityFunctors()[lossFunction]
        self.dataManager = dataManager
        self.isFit = False
         
       
    def addModel(self, name, model, fitting_arguments = {}):
        self.models[name] = model
        
        if(len(fitting_arguments) != 0) :
            self.fitArgs[name] = fitting_arguments
        

    def fit(self, scoring = None):

        if self.dataManager.isDataSplit == False :
            self.dataManager.splitData()
            
        if scoring is None :
            lossFunctor = self.lossFunctor
        else : lossFunctor = compositeDensityFunctors()[scoring]

            
        self.__fitImpl__(self.dataManager.X_train, self.dataManager.Y_train)
        
        for key, model in self.models.items():
            prediction = model.predict(self.dataManager.X_train) 
            losses = lossFunctor(prediction, self.dataManager.Y_train)
            self.trainResults[key] = _Results(losses)
        
        self.isFit = True


    def evaluate(self, scoring = None):
        
        if scoring is None :
            lossFunctor = self.lossFunctor
        else : lossFunctor = compositeDensityFunctors()[scoring]

        if self.isFit == False :
            print('comparator not fit yet')
            return 

        for key, model in self.models.items():
            prediction = model.predict(self.dataManager.X_test) 
            losses = lossFunctor(prediction, self.dataManager.Y_test)
            self.testResults[key] = _Results(losses)


    def __str__(self) :

        for key, value in self.models.items():
            print('results ' + key + ':')
            
            if key in self.trainResults : 
                self.trainResults[key].print('Train.MSE')
                    
            if key in self.testResults : 
                self.testResults[key].print('Test.MSE')
                
            if key in self.CV : 
                self.CV[key]['Total'].print('CV')
            
            print('')
                
        return ''
            
            
    def crossValidation(self, n_splits=5, n_repeats=1, random_state=2652124, scoring = None):
        
        #initialization
        if scoring is None :
            lossFunctor = self.lossFunctor
        else : lossFunctor = compositeDensityFunctors()[scoring]

        for key, model in self.models.items():
                self.CV[key] = {'Total' : _Results()}
      
        rkf = RepeatedKFold(n_splits, n_repeats, random_state)
        fold = 0
        
        #k.fold
        for train_index, test_index in rkf.split(self.dataManager.dataset.data):
            X_train, X_test = self.dataManager.dataset.data[train_index], self.dataManager.dataset.data[test_index]
            y_train, y_test = self.dataManager.dataset.target[train_index], self.dataManager.dataset.target[test_index]
            fold = fold + 1
            
            print('crossValidation@fold', fold, '...')
                 
            for key, model in self.models.items() :
                self.__fitImpl__(X_train, y_train)
                
                prediction = model.predict(X_test) 
                losses = lossFunctor(prediction, y_test)
                 
                self.CV[key][fold] = _Results(losses, train_index, model.get_params())
                self.CV[key]['Total'].append([self.CV[key][fold].mean])
                
        #calculate results
        for key, model in self.models.items() :
                self.CV[key]['Total'].calculate()
                
        print('CV done ...')      

           
        
    
    def __fitImpl__(self, X_train, y_train):

        for key, model in self.models.items():
            
            if key in self.fitArgs :
                args = self.fitArgs[key]

                for args_name, args_value in args.items():
                    print(args_name, ': ', args_value)

                args['x'] = X_train
                args['y'] = y_train
                
                model.fit(**args)
            else : 
                model.fit(X_train, y_train)

        
class _Results:
          
    def __init__(self, losses = None, train_index = None, train_parameters = None):
        
        self.train_index = train_index
        self.train_parameters = train_parameters
        self.losses = losses
        self.calculate()
            
    def append(self, losses):
        if self.losses is None :
            self.losses = losses
        else :
            self.losses = np.concatenate((self.losses, losses), axis=0)

        
    def calculate(self):
        if self.losses is not None :
            self.mean = np.mean(self.losses)
            self.std = np.std(self.losses)
            
    def print(self, tag):
        if tag == 'CV' :
            print(tag, ':', self.mean, '+-', self.std)
        else : print(tag, ':', self.mean)

if __name__ == "__main__" :
    print("class Comparator compile test ...")
