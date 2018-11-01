import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.layers import Dense
from sklearn.model_selection import train_test_split
# Importing the Dataset
dataset = pd.read_csv('Dataset/srp-class-csv.csv')
dataset['CLASS'] = (dataset['CLASS'] == False).astype(int)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
np.seterr(divide='ignore', invalid='ignore')
# Generate Dataset Function
def generateDataset(length, X, Y):
    DiX = []
    DiY = []
    for _ in range(length):
        p = np.random.randint(1, length)
        DiX.append(X[p])
        DiY.append(Y[p])
    return np.array(DiX),np.array(DiY)
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

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
    
    return accuracy,y_pred,classifier


def Logistic(X_train, X_test, y_train, y_test):
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0,0]+cm[1,1])/len(X_test)*100
    return accuracy,y_pred,classifier

def Knn(X_train, X_test, y_train, y_test):
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
   
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0,0]+cm[1,1])/len(X_test)*100
    return accuracy,y_pred,classifier

def Svm(X_train, X_test, y_train, y_test):
    classifier = SVC(kernel = 'poly',degree = 9, random_state = None)
    classifier.fit(X_train, y_train)
        
    y_pred = classifier.predict(X_test)  
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0,0]+cm[1,1])/len(X_test)*100
    return accuracy, y_pred,classifier

def RandomForest(X_train, X_test, y_train, y_test):
    classifier = RandomForestClassifier(n_estimators = 15, criterion = 'entropy', random_state = None)
    classifier.fit(X_train, y_train)
        
    y_pred = classifier.predict(X_test)  
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0,0]+cm[1,1])/len(X_test)*100
    return accuracy, y_pred,classifier

def DecisionTree(X_train, X_test, y_train, y_test):
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = None)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)  
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0,0]+cm[1,1])/len(X_test)*100
    return accuracy, y_pred,classifier


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
    return accuracy,y_pred,classifier


function = [RandomForest,Knn,DecisionTree,Svm,Logistic,NaiveBayes]
#[RandomForest,RandomForest,RandomForest,RandomForest,RandomForest,RandomForest,RandomForest]
f_name =['RandomForest','Knn','DecisionTree','Svm','Logistic','NaiveBayes']
for i in range(len(function)):
    functions =[function[0] for i in range(11)]
    print('\n',f_name[i])
    y_pred = []
    acc = []
    M = []
    for func in functions:
        accuracy,pred,m = func(x_train,x_test,y_train,y_test)
        y_pred.append(pred)
        acc.append(accuracy)
        M.append(m)
    model_train = []
    model_test = []
    for i in x_train:
        m_train=[]
        for j in range(0,len(functions)):
            m_train.append(M[j].predict(i.reshape(1,-1))[0])
        model_train.append(m_train)
    for i in x_test:
        m_test=[]
        for j in range(0,len(functions)):
            m_test.append(M[j].predict(i.reshape(1,-1))[0])
        model_test.append(m_test)
    stack_acc,y_preds,model = DecisionTree(model_train,model_test,y_train,y_test)
    print('Ind Acc : \n',acc,'\nStack Acc : ',stack_acc)
