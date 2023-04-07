import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split


def get_data():
    X = np.array(np.loadtxt('./ex2data1.txt',delimiter=','))
    y = X[:,-1]

    x = X[0:,:-1]
    x_tr, x_te, y_tr, y_te = train_test_split(x,y,test_size=0.3)
    x_tr = np.hstack((x_tr, np.ones((x_tr.shape[0], 1))))
    x_te = np.hstack((x_te, np.ones((x_te.shape[0], 1))))
    return x_tr,x_te,y_tr,y_te


def get_bc():
    xy = datasets.load_breast_cancer()
    x = xy['data']
    y = xy['target']
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2)
    x_tr = np.hstack((x_tr, np.ones((x_tr.shape[0], 1))))
    x_te = np.hstack((x_te, np.ones((x_te.shape[0], 1))))
    return x_tr, x_te, y_tr, y_te

