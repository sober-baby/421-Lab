import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix 

def fit_perceptron(X_train, y_train):

    # add bias
    X = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    # create weight vector
    w = np.zeros(X.shape[1])
    
    #initialize best weight and error (to be replaced later during training)
    best_w = w.copy()
    best_error = errorPer(X, y_train, w)  
    
    #define number of epochs
    max_epochs = 5000
    
    #go through each epoch
    for epoch in range(max_epochs):
        # go through each sample
        for i in range(len(y_train)):
            #get the prediction
            y_pred = pred(X[i], w)           
            if y_pred != y_train[i]:
                w = w + y_train[i] * X[i]
                #check if this w is better and store it if it is
                current_error = errorPer(X, y_train, w)
                if current_error < best_error:
                    best_error = current_error
                    best_w = w.copy()
    
    return best_w

    
    

def errorPer(X_train,y_train,w):
    misclassified = 0
    # go through each sample
    for i in range(len(y_train)):
        # determine the number of miscalssfied points
        if pred(X_train[i], w) != y_train[i]:
            misclassified += 1
    avgError = misclassified / len(y_train)
    return avgError
    

def confMatrix(X_train,y_train,w):
    X = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    
    # initialize confusion matrix
    confu = np.zeros((2, 2), dtype=int)
    
    for i in range(len(y_train)):
        actual = y_train[i]
        predicted = pred(X[i], w)
        if actual == -1:
            if predicted == -1:
                confu[0, 0] += 1  # TN
            else:
                confu[0, 1] += 1  # FP
        else: 
            if predicted == -1:
                confu[1, 0] += 1  # FN
            else:
                confu[1, 1] += 1  # TP
    return confu
 

def pred(X_i,w):
    #predicts the class label
    if np.dot(w, X_i) > 0:
        return 1
    else:
        return -1

def test_SciKit(X_train, X_test, Y_train, Y_test):
    # api call
    p = Perceptron(max_iter=5000, tol=None)
    p.fit(X_train, Y_train)
    
    # obtain prediction
    y_pred = p.predict(X_test)
    
    return confusion_matrix(Y_test, y_pred)
 

def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2)
    
    # print(X_train)

    y_train[y_train == 1] = 1
    y_train[y_train != 1] = -1
    y_test[y_test == 1] = 1
    y_test[y_test != 1] = -1

    w=fit_perceptron(X_train,y_train)
    confu=confMatrix(X_test,y_test,w)

    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print ('--------------Test Result-------------------')
    print("Confusion Matrix is from Part 1a is: ",confu)
    print("Confusion Matrix from Part 1b is:",sciKit)
    

test_Part1()
