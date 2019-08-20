# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

class dataManager:
    
    def __init__(self, dataSet,  test_size = 0.2, random_state=5, normalize_data=True):
        
        self.crossValResults = {}
        self.dataset = dataSet
        
        self.Y_train = {}
        self.X_train = {}
        self.Y_test = {}
        self.X_test = {}

        self.test_size = test_size
        self.random_state = random_state
        self.normalize_data = normalize_data
        
        self.isDataSplit = False
        self.isFit = False
        
        
    def splitData(self, test_size = None, random_state= None, normalize_data = None) :

        if test_size is not None :
            self.test_size = test_size
            
        if random_state is not None :
            self.random_state = random_state
            
        if normalize_data is not None :
            self.normalize_data = normalize_data
            
        X_train, X_test, Y_train, Y_test = train_test_split(
                self.dataset.data, self.dataset.target, 
                test_size = self.test_size, 
                random_state = self.random_state)
        
        if normalize_data == True :
            X_train = normalize(X_train)
            X_test = normalize(X_test)

        self.Y_train = Y_train
        self.X_train = X_train
        self.Y_test = Y_test
        self.X_test = X_test
        self.isDataSplit = True
       
