from sklearn import preprocessing

import sklearn 
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head()) #To check if our data is loaded correctly 

#Converting Data
le = preprocessing.LabelEncoder()

#Take a list and return it an array containing the values
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))


X = list(zip(buying, maint, door, persons, lug_boot, safety)) #features
y = list(cls) #labels


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
