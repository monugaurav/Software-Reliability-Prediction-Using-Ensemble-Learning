# Task1 of Project

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.layers import Dense
#import xlwt
#from xlwt import Workbook

# Importing the dataset
dataset = pd.read_csv('Dataset/srp-class-csv.csv')
dataset['CLASS'] = (dataset['CLASS'] == False).astype(int)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

"""
# Encoding the Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
dataset1[:, 1] = labelencoder_X_1.fit_transform(dataset1[:, 1])
labelencoder_X_2 = LabelEncoder()
dataset1[:, 2] = labelencoder_X_2.fit_transform(dataset1[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
dataset1 = onehotencoder.fit_transform(dataset1).toarray()
dataset1 = dataset1[:, 1:]
#print(len(dataset1))"""

# Feature Scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


def generateDataset(length, X, Y):
    DiX = []
    DiY = []
    for _ in range(length):
        p = np.random.randint(1, length)
        DiX.append(X[p])
        DiY.append(Y[p])
    return np.array(DiX),np.array(DiY)


def NaiveBayes(X_train, X_test, y_train, y_test):
    '''sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    # Predicting the Test set results
    X_test = sc.transform(X_test)'''
    
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0,0]+cm[1,1])/len(X_test)*100
    
    return accuracy,y_pred


def Logistic(X_train, X_test, y_train, y_test):
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0,0]+cm[1,1])/len(X_test)*100
    return accuracy,y_pred

def Knn(X_train, X_test, y_train, y_test):
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
   
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0,0]+cm[1,1])/len(X_test)*100
    return accuracy,y_pred

def Ann(X_train, X_test, y_train, y_test):
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    # Adding the second hidden layer
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    # Adding the output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100, verbose=0)
    # Part 3 - Making the predictions and evaluating the model

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int).ravel()
   

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0,0]+cm[1,1])/len(X_test)*100
    return accuracy,y_pred

def Svm(X_train, X_test, y_train, y_test):
    classifier = SVC(kernel = 'poly',degree = 9, random_state = None)
    classifier.fit(X_train, y_train)
        
    y_pred = classifier.predict(X_test)  
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0,0]+cm[1,1])/len(X_test)*100
    return accuracy, y_pred

def RandomForest(X_train, X_test, y_train, y_test):
    classifier = RandomForestClassifier(n_estimators = 15, criterion = 'entropy', random_state = None)
    classifier.fit(X_train, y_train)
        
    y_pred = classifier.predict(X_test)  
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0,0]+cm[1,1])/len(X_test)*100
    return accuracy, y_pred

def DecisionTree(X_train, X_test, y_train, y_test):
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = None)
    classifier.fit(X_train, y_train)
        
    y_pred = classifier.predict(X_test)  
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0,0]+cm[1,1])/len(X_test)*100
    return accuracy, y_pred


def ann(X_train, X_test, y_train, y_test):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu', input_dim = len(X_train[0])))
    # Adding the second hidden layer
    classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'tanh'))
    
    classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'tanh'))
    
    classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'tanh'))
    
    classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'tanh'))
    # Adding the output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100, verbose = 0)
    # Part 3 - Making the predictions and evaluating the model

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int).ravel()
   
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0,0]+cm[1,1])/len(X_test)*100
    return accuracy,y_pred


functions = [ann,RandomForest,Knn,DecisionTree,Svm,Logistic,NaiveBayes]
f_name =['Ann','RandomForest','Knn','DecisionTree','Svm','Logistic','NaiveBayes']
bagging_accuracy = {}
k = 3
def bagging(k=5):
    for p in range(len(functions)):
        #k = 5
        count = 0
        _sum = 0
        acc = {}
        pred = []
        for i in range(k):
            dix = []
            diy = []
            dix, diy = generateDataset(len(X), X, y)
            acc[i],b = functions[p](dix, X,diy, y)
            pred.append(b)
        bag = []
        for i in range(len(pred[0])):
            x = []
            for j in range(len(pred)):
                x.append(pred[j][i])
            bag.append(max(set(x),key=x.count))
        
        max_key = max(acc.keys(), key=(lambda k: acc[k]))
        min_key = min(acc.keys(), key=(lambda k: acc[k]))
        
        max_val = acc[max_key]
        min_val = acc[min_key]
        
        for key in acc:
            count += 1
            _sum += acc[key]
        avg_val = _sum/count
        
        
        cm = confusion_matrix(y, bag)
        accuracy = (cm[0,0]+cm[1,1])/len(X)*100
        bagging_accuracy[f_name[p]] = accuracy
        #print('\n',f_name[p],'\n',acc,'\n',accuracy,'\n',cm)
        print('======================================')
        print('\n','Method Name:',f_name[p],'\n')
        '''print('K: ',k,'\n')
        print('Max: ',max_val,'\n')
        print('Min: ',min_val,'\n')'''
        print('Avg: ',avg_val,'\n')
        #print('Accuray: ','\n',acc,'\n')
        print('Bagging Accuracy: ',accuracy,'\n')
        print('Confusion Matrix: ','\n',cm,'\n')
'''ran = int(input("enter range(odd num only): "))
while(k <= ran):
    
    k += 2'''
bagging(k=5)
    
