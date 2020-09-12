# Air Pollution Prediction using Mutivariate Linear Regression
# @author - Kunal Jindal

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Getting Data

df_test = pd.read_csv('Test.csv')
df_train = pd.read_csv('Train.csv')


df_train.columns
print(type(df_train))

X = df_train.values[:,:5]
Y = df_train.values[:,-1]
X_test = df_test.values
print(type(X))


print(X.shape)
print(Y.shape)
print(X_test.shape)


# Normalisation

u = np.mean(X,axis=0)
std = np.std(X,axis=0)
X = (X-u)/std

u = np.mean(X_test,axis=0)
std = np.std(X_test,axis=0)
X_test = (X_test-u)/std



# Linear Regression 

# Appending extra column of 1
ones = np.ones((X.shape[0],1))
X = np.hstack((ones,X))
print(X.shape)

ones = np.ones((X_test.shape[0],1))
X_test = np.hstack((ones,X_test))
print(X_test.shape)

m = 1600


#Algorithm
def hypothesis(X,theta):
    return np.dot(X,theta)

def error(X,y,theta):
    e = 0.0
    y_ = hypothesis(X,theta)
    e = np.sum((y-y_)**2)
    
    return e/m

def gradient(X,y,theta):
    y_ = hypothesis(X,theta)
    grad = np.dot(X.T,(y_-y))
    m = X.shape[0]
    return grad/m

def gradient_descent(X,y,learning_rate=0.1,max_iters=300):
    n = X.shape[1]
    theta = np.zeros((n,))
    error_list = []
    
    for i in range(max_iters):
        e = error(X,y,theta)
        error_list.append(e)
        
        #Gradient Descent
        grad = gradient(X,y,theta)
        theta = theta - learning_rate*grad
        
    return theta,error_list


#Training
theta,error_list = gradient_descent(X,Y)

#Testing
y_ = hypothesis(X_test,theta)

#Final Conversion to CSV File

df = pd.DataFrame({'target': y_})
df.to_csv('submission.csv',index=True,index_label='Id')


# Calculating r2 score
def r2_score(y,y_):
    num = np.sum((y-y_)**2)
    denom = np.sum((y-y.mean())**2)
    score = (1-num/denom)
    return score*100

r2_score(y_,Y)






