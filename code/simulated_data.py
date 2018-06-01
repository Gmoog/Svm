"""
Script to generate simulated data and run the SVM algorithm on it. 
The data is generated using np.random.normal and np.random.uniform functions.
It is a simple dataset consisting of 100 observations and 10 features.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import Svm

# Generate a dataset with 100 obervations and 10 features
np.random.seed(100)
features = np.zeros((100, 10))
features[0:50, :] = np.random.normal(loc = 2, scale=3, size=(50, 10))
features[50:100, :] = np.random.uniform(low =1, high=5, size=(50, 10))
# Generate labels with half as 1 and rest as -1
labels = np.asarray([1]*50 + [-1]*50)

# Random train-test split. Test set contain 75% of the data.
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=0)

# Standardize the data
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Run the algorithm for Lambda = 1
svm = Svm.Svm(max_iter = 50)
lambduh = 1
d = x_train.shape[1]
beta_init = np.zeros(d)
theta_init = np.zeros(d)
eta_init = 1/(scipy.linalg.eigh(1/len(y_train)*x_train.T.dot(x_train), eigvals=(d-1, d-1), eigvals_only=True)[0]+lambduh)
betas_fastgrad= svm.mylinearsvm(beta_init, theta_init, lambduh, x_train.T, y_train, eta_init)
print('Misclassification error on training data when lambda=1:', svm.classification_error(betas_fastgrad[-1],x_train.T,y_train))
print('Misclassification error on test data when lambda=1:', svm.classification_error(betas_fastgrad[-1],x_test.T,y_test))
svm.objective_plot(betas_fastgrad, lambduh, x_train.T,y_train)

# Run crossvalidation to find optimal lambda value
lambdah = 32
lamlist = []
for i in range(30):
	lamlist.append(lambdah)
	lambdah /= 2
opt_l,err = svm.crossvalidation(5,x_train.T,y_train,lamlist,beta_init,theta_init,eta_init)
print('Optimal value of lambda using 5 fold Cv is:', opt_l)
opt_lam,errc = svm.lamda_min_error(lamlist,x_train.T,y_train,beta_init,theta_init,eta_init)
print('Misclassification error on optimal lambda value is:' , errc)
