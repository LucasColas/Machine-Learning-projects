import sklearn
from sklearn import svm
from sklearn import datasets

cancer = datasets.load_breast_cancer()

print("Features: ", cancer.feature_names)
print("Labels: ", cancer.target_names)

