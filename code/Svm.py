"""
The python class Svm contains various functions to implement the SVM model using the fast gradient descent algorithm.
The backtracking rule is used to speed up the beta optimization.

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import KFold
import itertools


class Svm:
    
    def __init__(self, max_iter=100):
        self.max_iter = max_iter


    # Function to calculate gradient
    def gradient(self, beta, lamda, x, y):
        b = beta
        l = lamda
        n = y.shape[0]
        yx = y[:,np.newaxis]*x.T
        loss = np.maximum(0,1 - yx@b)
        return (-2/n)*loss@yx + 2*l*b


    # Function to calculate cost
    def objective(self, beta, lamda, x, y):
        b = beta
        l = lamda
        n = y.shape[0]
        yx = y[:,np.newaxis]*x.T
        loss = np.maximum(0,1 - yx@b)
        return 1/n * (np.sum(loss**2)) + (l * (np.linalg.norm(b)**2))


    # function to implement the backtracking rule    
    def backtracking(self, b, lamda, x, y, t, alpha=0.5, thet=0.8):
        grad_b = self.gradient(b,lamda,x,y)
        norm_grad_b = np.linalg.norm(grad_b)
        found_t = False
        i = 0
        while(found_t is False) and i < self.max_iter:
            if (self.objective(b-t*grad_b,lamda,x,y) < self.objective(b,lamda,x,y) - alpha*t*norm_grad_b**2):
                found_t = True
            else:
                t *= thet
            i += 1
        return t

    # function to implement fast gradient algorithm
    def mylinearsvm(self, beta, theta, lamda, x, y, t_init):
        b_vals = [beta]
        i = 0
        grad_b = self.gradient(beta,lamda,x,y)
        while i < self.max_iter:
            t = self.backtracking(beta,lamda,x,y,t_init)
            beta = theta - t*self.gradient(theta,lamda,x,y)
            b_vals.append(beta)
            theta = beta + i/(i+3)*(beta - b_vals[-2])
            grad_b = self.gradient(beta,lamda,x,y)
            i += 1
        return b_vals

    # function to calculate the classification error
    def classification_error(self, bval, x, y):
        prediction = x.T @ bval
        prediction_bool = np.array([1 if x > 0 else -1 for x in prediction])
        return np.mean(prediction_bool != y)

    # function to perform cross validation to select optimal regularization parameter
    def crossvalidation(self, k, x, y, lamlist, be, th, eta):
        lam = lamlist
        x = x.T
        y = np.array(y)
        mean_err = []
        kf = KFold(n_splits=k)
        for l in lam:
            error = []
            for train,test in kf.split(x):
                bt = []
                x_tra, x_te = x[train], x[test]
                y_tra, y_te = y[train], y[test]
                y_tra = pd.Series(y_tra)
                y_te = pd.Series(y_te)
                bt = self.mylinearsvm(be,th,l,x_tra.T,y_tra,eta)
                error.append(self.classification_error(bt[-1],x_te.T,y_te))
            mean_err.append(np.mean(error))
        return lam[np.argmin(mean_err)],np.min(mean_err)

    # function to find optimal lambda using classification error
    def lamda_min_error(self,llist,x,y,be,th,eta):
        mis_error = np.zeros(len(llist))
        for i,j in enumerate(llist):
            betas = self.mylinearsvm(be,th,j,x,y,eta)
            mis_error[i] = self.classification_error(betas[-1],x,y) 
            i= i+1
        return llist[np.argmin(mis_error)], np.min(mis_error)

    # function to plot the objective value vs lambda
    def objective_plot(self, betas_fg, lambduh, x, y):
        num_points = np.size(betas_fg, 0)
        objs_fg = np.zeros(num_points)
        for i in range(0, num_points):
            objs_fg[i] = self.objective(betas_fg[i], lambduh, x, y)
        fig, ax = plt.subplots()
        ax.plot(range(1, num_points + 1), objs_fg)
        plt.xlabel('Iteration')
        plt.ylabel('Objective value')
        plt.title('Objective value vs. iteration when lambda='+str(lambduh))
        plt.show()
        




