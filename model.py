#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
#import dataset
dataset = pd.read_csv('sales.csv')

dataset['rate'].fillna(0, inplace=True)

dataset['sales_in_first_month'].fillna(dataset['sales_in_first_month'].mean(), inplace=True)

#split data into train and test
X = dataset.iloc[:, :3]

def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['rate'] = X['rate'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

#use linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X, y)

#by creating model.pkl you are dumping all the data in the regressor model
#into the pickle dump
pickle.dump(regressor, open('model.pkl','wb')) #wb=write mode

model = pickle.load(open('model.pkl','rb')) #rb=read mode
print(model.predict([[4, 300, 500]]))