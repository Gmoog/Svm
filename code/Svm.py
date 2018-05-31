# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
import itertools


class Svm:
    
    def __init__(self, max_iter=100):
        self.max_iter = max_iter


    def standardize_data(self, x_path, y_path, class1, class2):
        test = np.load(x_path)
        test_lab = np.load(y_path)
        df = pd.DataFrame(test)
        df1 = pd.DataFrame(test_lab)
        test_df = pd.concat([df,df1],axis = 1,ignore_index = True)
        # choosing 2 classes from the dataset
        temp1 = test_df[test_df.iloc[:,-1] == class1]
        temp2 = test_df[test_df.iloc[:,-1] == class2]
        test_class = pd.concat([temp1,temp2], axis = 0)
        test_class.iloc[:,-1] = test_class.iloc[:,-1].map({class1:-1,class2:1})
        X = (test_class.loc[:,:4095]).T
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
        y = test_class.iloc[:,-1]
        return X,y

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
        clas_error = []
        grad_b = self.gradient(beta,lamda,x,y)
        while i < self.max_iter:
            t = self.backtracking(beta,lamda,x,y,t_init)
            beta = theta - t*self.gradient(theta,lamda,x,y)
            er = self.classification_error(beta,x,y)
            clas_error.append(er)
            b_vals.append(beta)
            theta = beta + i/(i+3)*(beta - b_vals[-2])
            grad_b = self.gradient(beta,lamda,x,y)
            i += 1
        # return b_vals instead of clas_error to perform cross validation
        return clas_error

    # function to calculate the classification error
    def classification_error(self, bval, x, y):
        prediction = x.T @ bval
        prediction_bool = np.array([1 if x > 0.5 else -1 for x in prediction])
        return np.mean(prediction_bool != y)

    # function to perform cross validation to select optimal regularization parameter
    def crossvalidation(self, k, x, y, be, th, eta):
        lam = [0.15,0.1,0.05,0.5,1]
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
                #print("inside cv loop")
                bt = self.mylinearsvm(be,th,l,x_tra.T,y_tra,eta)
                error.append(self.classification_error(bt[-1],x_te.T,y_te))
            mean_err.append(np.mean(error))
        return lam[np.argmin(mean_err)],np.min(mean_err)

    def one_v_one(self):
        pairs = list(itertools.combinations(range(0,10),2))
        df = pd.DataFrame(pairs,columns=['Class1','Class2'])
        opt_lam = []
        ms_err = []
        for i,j in pairs:
            print(i,j)
            X_train,y_train = self.standardize_data("train_features.npy","train_labels.npy",i,j)
            beta_init = np.zeros(X_train.shape[0])
            theta_init = np.zeros(X_train.shape[0])
            eta_init = 20
            l,err = self.crossvalidation(3,X_train,y_train,beta_init,theta_init,eta_init)
            opt_lam.append(l)
            ms_err.append(err)
        df['Crossval_Reg_Parameter'] = opt_lam
        df['Class_Error'] = ms_err
        df.to_csv('one_v_one_table.csv')

    def get_betas(self,lamda):
        pairs = list(itertools.combinations(range(0,10),2))
        betas = []
        for i,j in pairs:
            print(i,j)
            X_train,y_train = self.standardize_data("train_features.npy","train_labels.npy",i,j)
            beta_init = np.zeros(X_train.shape[0])
            theta_init = np.zeros(X_train.shape[0])
            eta_init = 20
            bet = self.mylinearsvm(beta_init,theta_init,l,X_train,y_train,eta_init)
            betas.append(bet[-1])
        data_f = pd.DataFrame(betas)





if __name__=='__main__':
    svm = Svm(max_iter = 100)
    X_train,y_train = svm.standardize_data("train_features.npy","train_labels.npy",1,8)
    X_val,y_val = svm.standardize_data("val_features.npy","val_labels.npy",1,8)
    beta_init = np.zeros(X_train.shape[0])
    theta_init = np.zeros(X_train.shape[0])
    lamda = 0.1
    eta_init = 20
    #svm.one_v_one()
    # comment code to perform cross validation
    class_error_train = svm.mylinearsvm(beta_init,theta_init,lamda,X_train,y_train,eta_init)
    class_error_train = pd.DataFrame(class_error_train)
    class_error_train.to_csv('error_training_cv.csv',header=['Error'])
    class_error_val = svm.mylinearsvm(beta_init,theta_init,lamda,X_val,y_val,eta_init)
    class_error_val = pd.DataFrame(class_error_val)
    class_error_val.to_csv('error_val_cv.csv',header=['Error'])
    #uncomment code to perform crossvalidation
    #optim_lambda,ms_err = svm.crossvalidation(3,X_train,y_train,beta_init,theta_init,eta_init)
    #print("Optimal value of lamda using 3 fold CV is: ", optim_lambda, "and the misclassification error is: ", ms_err)