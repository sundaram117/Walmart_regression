#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
import datetime
from sklearn.base import BaseEstimator
import pdpipe as pdp

# class definition for joining multiple model throuh pipes
class ClfSwitcher(BaseEstimator):

    def __init__(
        self, 
        estimator = LinearRegression(),
    ):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """ 

        self.estimator = estimator


    def fit(self, X, y = None, **kwargs):
        self.estimator.fit(X, y)
        return self


    def predict(self, X, y = None):
        return self.estimator.predict(X)


    def score(self, X, y):
        return self.estimator.score(X, y)






def preprocessing_walmart(dataset):# 2nd function definition  processing
    dataset['Day'] = pd.to_datetime(dataset['Date'])

    #using pandas pipeline
    panda_pipe = pdp.ApplyByCols('Day',lambda x: (x.day//7)+1,'Week_no',drop = False)
    #converting given day into week of the month

    panda_pipe += pdp.ApplyByCols('Day',lambda x: x.month,'month',drop = False)
    #getting month from the date

    panda_pipe += pdp.ColDrop(['Date','Day'])
    dataset = panda_pipe(dataset)
   
    dataset['Lag2'] = dataset['Weekly_Sales'].shift(2)
    dataset['Lag3'] = dataset['Weekly_Sales'].shift(3)
    dataset['Lag4'] = dataset['Weekly_Sales'].shift(4)
    dataset['Lag5'] = dataset['Weekly_Sales'].shift(5)
    dataset['Lag6'] = dataset['Weekly_Sales'].shift(6)
    

    to_be_predicted = dataset['Weekly_Sales']
    dataset = dataset.drop(columns = ['Weekly_Sales'])
    X_train,X_test,Y_train,Y_test = train_test_split(dataset, to_be_predicted, random_state = 42, test_size = 0.3)

    return (X_train,Y_train,X_test)





def model_pipeline():#3rd function definition

	#referenced class ClfSwitcher for optimal parameter
	pipe = Pipeline([('standard',StandardScaler()),('reg',ClfSwitcher())])
	'''
	class reference for selecting the best models
	'''

	params = [{'reg__estimator':[LinearRegression()]},
	       {
	        'reg__estimator':[Ridge()],
	       'reg__estimator__alpha':[0.1,1,10]},
	       {
	        'reg__estimator':[Lasso()],
	           'reg__estimator__alpha':[.001,.01,.1,1,10]
	       }]

	#given evaluation states to pipe(ClfSwitcher(BaseEstimator=LinearRegression))

	#using gridsearchCV for best model selection
	gscv = GridSearchCV(pipe, params, cv=5)
	return gscv


def results(X_train, Y_train, X_test, model):
    model.fit(X_train,Y_train)
    model.predict(X_test)


def main():# First Function definition main
    dataset = pd.read_csv("/home/sundaram/walmart-recruiting-store-sales-forecasting/train.csv")
    X_train,Y_train,X_test = preprocessing_walmart(dataset)# 2nd function call for processing
    model=model_pipeline()
    results(X_train, Y_train, X_test, model)

if __name__ == "__main__":
	main()