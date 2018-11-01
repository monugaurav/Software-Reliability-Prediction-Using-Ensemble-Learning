# Prediction System

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoLarsCV 
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LogisticRegression


def lag(x,k):
    lag_len = len(x)
    z = [list(x[i:i+k].reshape(-1)) for i in range(0, lag_len-k+1)]
    return( np.array(z))

def generateDataset(length, X, Y):
    DiX = []
    DiY = []
    for _ in range(length):
        p = np.random.randint(1, length)
        DiX.append(X[p])
        DiY.append(Y[p])
    return np.array(DiX),np.array(DiY)
# Importing the Datasets
dataset = pd.read_csv('Dataset/srp-pred-csv.csv')
d = dataset.iloc[:, 1].values
org_data = np.array(d)
d = np.array(d)
for i in range(1,len(d)):
    d[i] = d[i]+d[i-1]
d = lag(d,10)
X = d[:,:-1]
y = d[:,-1]

def svr(X,y,value):
    regressor = SVR(kernel = 'rbf',gamma='scale')
    regressor.fit(X, y)
    y_pred = regressor.predict(value)
    return y_pred
    
def randomForest(X,y,value):
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(X,y)
    y_pred = regressor.predict(value)
    return y_pred
    
    
def decisonTree(X,y, value):
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X, y)
    y_pred = regressor.predict(value)
    return y_pred

def ridge(X,y,value):
    regressor = Ridge(alpha = 1)
    regressor.fit(X,y)
    y_pred = regressor.predict(value)
    return y_pred

def lasso(X,y,value):
    regressor = LassoLarsCV(cv = 10, precompute = False)
    regressor.fit(X,y)
    y_pred = regressor.predict(value)
    return y_pred

def elasticNet(X,y,value):
    regressor = ElasticNet(random_state = None,alpha=0.3,max_iter=5000)
    regressor.fit(X,y)
    y_pred = regressor.predict(value)
    return y_pred

def linearRegression(X, y, value):
    regressor = LinearRegression()
    regressor.fit(X, y)
    y_pred = regressor.predict(value)
    return y_pred

def bayesianRidge(X, y, value):
    regressor = BayesianRidge()
    regressor.fit(X, y)
    y_pred = regressor.predict(value)
    return y_pred

def lassoLars(X, y, value):
    regressor = LassoLars(alpha = 0.3,max_iter=600000)
    regressor.fit(X, y)
    y_pred = regressor.predict(value)
    return y_pred

def logisticRegression(X, y, value):
    regressor = LogisticRegression(random_state = 0,solver='liblinear',max_iter=5000,multi_class='auto')
    regressor.fit(X, y)
    y_pred = regressor.predict(value)
    return y_pred

print('')

x = [i for i in range(1,1+len(y))]
functions = [ridge, lasso,  decisonTree, randomForest, svr, linearRegression,
             bayesianRidge, lassoLars, logisticRegression]
f_name = ['ridge', 'lasso','decisonTree', 'randomForest', 'svr', 'linearRegression'
          , 'bayesianRidge','lassoLars','logisticRegression']
for p in range(len(functions)):
    k = 5
    pred = []
    #average = []
    for i in range(k):
        dix = []
        diy = []
        dix, diy = generateDataset(len(X), X, y)
        b = functions[p](dix,diy, X)
        pred.append(b)
    pred = np.array(pred)
    avg = pred.mean(axis=0)
    plot_y = np.array(y)
    plot_avg = np.array(avg)
    for i in range(len(y)-1,1,-1):
        plot_y[i] = plot_y[i]-plot_y[i-1]
        plot_avg[i] = plot_avg[i]-plot_avg[i-1]
        
    plt.clf()
    plt.xlim(1,367)
    plt.ylim(min(plot_y), max(plot_y))
    
    plt.plot(x,plot_y,color='green',marker='.')
    plt.plot(x,plot_avg,color='red',marker='x')
    plt.show()
    plt.savefig(''+f_name[p]+'.png')
    
    w = np.array(y)
    avg = np.array(avg)
    w = (w - w.min()) / (w.max() - w.min())
    avg = (avg - avg.min()) / (avg.max() - avg.min())
    
    print('Error of ',f_name[p], ' : ', format(float(mean_squared_error(w, avg)), 'f'))
