# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

dataset = pd.read_csv("hiring.csv")

dataset["test_score"].fillna(0, inplace = True)

X = dataset.iloc[:, :3]

def convet_to_num(w):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven ':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, '0': 0}
    return word_dict[w]

X['experience'] = X['experience'].apply(lambda x: convet_to_num(x))

print(X)
y = dataset.iloc[:,-1]
print(y)

regressor = LinearRegression()

regressor.fit(X,y)

pickle.dump(regressor, open("model.pkl", 'wb'))

mo = pickle.load(open('model.pkl','rb'))

print(mo.predict([[0,8.0,9]]))
