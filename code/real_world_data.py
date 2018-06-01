"""
This function is used to run the svm algorithm on a real world dataset.
The dataset used is the Spam dataset from the book, elements of statistical learning.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import sklearn.preprocessing
import Svm

# Get the data from the Spam Dataset
spam = pd.read_table('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data', sep=' ', header=None)
test_indicator = pd.read_table('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.traintest', sep=' ',header=None)

# Store the features and labels as an array
x = np.asarray(spam)[:, 0:-1]
y = np.asarray(spam)[:, -1]*2 - 1 

# Use the train-test split inidcator provided along with the dataset
test_indicator = np.array(test_indicator).T[0]
x_train = x[test_indicator == 0, :]
x_test = x[test_indicator == 1, :]
y_train = y[test_indicator == 0]
y_test = y[test_indicator == 1]

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
print('Optimal value of lambda using 5 fold Cv is:', opt_l, 'and the mean error is :', err)
opt_lam,errc = svm.lamda_min_error(lamlist,x_train.T,y_train,beta_init,theta_init,eta_init)
print('Optimal value of lambda using classification error is:', opt_lam)
print('Misclassification error on optimal lambda value is:' , errc)



