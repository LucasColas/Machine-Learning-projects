import pandas as pd 
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle 
import matplotlib.pyplot as plt 
import pickle

data = pd.read_csv("student-mat.csv", sep=";")

#print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]] #trimming our data

predict = "G3"

X = np.array(data.drop([predict], 1)) #Features
Y = np.array(data[predict]) 

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size= 0.1) #90 % of our data to train and the other 10% to test


linear = linear_model.LinearRegression() #Define the model

linear.fit(x_train, y_train)

acc = linear.score(x_test, y_test)

with open("studentmodel.pickle", "wb") as f:
	pickle.dump(linear, fx)


print("accuracy : ")
print(acc) #accuracy 

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
	print(predictions[x], x_test[x], y_test[x])

