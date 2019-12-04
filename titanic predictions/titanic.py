import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel('titanic3.xls')

#data.shape
#data.columns
#data.head()

data = data.drop(['name', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest'], axis=1)

data.head()



