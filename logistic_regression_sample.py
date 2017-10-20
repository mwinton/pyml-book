#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 21:41:00 2017

@author: mike
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


class Perceptron(object):
    """Perceptron classifier.
    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.
    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.
        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        # This is gradient descent, but keeps running through n_iter no matter what
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)



class LogisticRegressionGD(object):
    """Logistic Regression classifier with gradient descent
    
    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    alpha : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.
      
    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum of squares cost function value in each epoch.
    """
    
    def __init__(self, alpha = 2.0, eta=0.05, n_iter=100, random_state=1, to_normalize = True):
        self.eta = eta
        self.alpha = alpha
        self.n_iter = n_iter
        self.random_state = random_state
        self.to_normalize = to_normalize
        if self.to_normalize:
            self.X_means = []
            self.X_steds = []


    def fit(self, X, y):
        """
        Fits training data.  Optimizes w_ with gradient descent.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.
    
        Returns
        -------
        self : object
        """
 
        if self.to_normalize:
            (X, self.X_means, self.X_stds) = self.normalize(X)    
            # If we chose to normalize data, re-plot it.
            plt.scatter(X[:50,0], X[:50,1], color='red', marker='o', label='Setosa')
            plt.scatter(X[50:100,0], X[50:100,1], color='blue', marker = 'x', label='Virginica')
            plt.xlabel('sepal length [cm]')
            plt.ylabel('petal length [cm]')
            plt.title('Iris data set (normalized)')
            plt.legend()
            plt.show()
       
        # Set up initial values for weights
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []
        m = len(y) # Andrew Ng definition

        #Run gradient descent (NOTE: it keeps running through n_iter no matter what)
        for i in range(self.n_iter):
            net_input = self.net_input(X) #scalar
            output = self.sigmoid(net_input) #vector
            errors = (y - output) #vector
#            self.w_[0] += self.eta * errors.sum() #vector; w_[0] is bias unit
#            self.w_[1:] += self.eta * X.T.dot(errors) #vector
            self.w_[0] += (self.alpha / m) * errors.sum() #vector; w_[0] is bias unit
            self.w_[1:] += (self.alpha / m)  * X.T.dot(errors) #vector
            cost = self.cost_function(y, output)
            self.cost_.append(cost) #used to verify convergence
        return self

        
    def normalize(self,X):
        X_norm = np.copy(X)
        X_means = []
        X_stds = []
        num_cols = X_norm.shape[1]
        
        for i in range(0,num_cols):
            X_norm[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()

            #Store normalization factors to restore predicted values later
            X_means.append(X[:,i].mean())
            X_stds.append(X[:,i].std())
        print('\nDuring normalization, calculated mean=',X_means,'std=',X_stds)

        return(X_norm, X_means, X_stds)
        
        
    def net_input(self, X):
        """Calculate net input.  This term is z in Andrew Ng's class """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def sigmoid (self, z):
        '''Calculate sigmoid function fron net_input (z)'''     
        # z values > 250 (or < -250) are clipped at 250
        return 1. / (1. + np.exp(-np.clip(z,-250,250)))
        
    def cost_function (self,y,output):
        '''Calculate the logistic regression cost function'''
        m = len(y)
        cost = (1/m) * (-y.dot(np.log(output)) - ((1.-y).dot(np.log(1.-output))))
        return cost
            
        
    def predict(self, X):
        """Return class label based on sigmoid function"""
        
        if self.to_normalize:
            X_norm = np.copy(X)
            for i in range(0,X_norm.shape[1]):
 #               print('Normalizing to do prediction.',X_means[i],X_stds[i])
                # During prediction, need to use the saved means/stds, rather than recalculate
                X_norm[:,i] = (X[:,i] - self.X_means[i]) / self.X_stds[i]
            return np.where(self.net_input(X_norm) >= 0, 1, 0)
        else:
            #Due to shape of sigmoid function, these two options are equivalent
            #return np.where(self.net_input(X) >= 0, 1, 0)
            return np.where(self.sigmoid(self.net_input(X)) >= 0.5, 1, 0)
 



def plot_decision_regions(X, y, classifier, resolution=0.02):
    '''
    Helper function to plot the decision regions.
    Data does not need to be pre-normalized. 
    Classifier's predict() function will noramlize if necessary, but
    data will be plotted using original values.
    '''
    
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
 
def loadDataset(model = 'lr'):
    '''
    Loads data from UCI repository.  
        Transforms y in [0,1] for model = 'lr' (default)
        Transforms y in [-1,1] for model = 'p'
    '''
    
    df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    #df = pd.read_csv('local/path/to/data', header=None)
    print(df.tail())
    
    # select setosa and verginica (using only 100 of the 150 samples for training)
    y = df.iloc[0:100, 4].values
    
    # deal with the fact that perceptron treats negative cases as -1; LR treats as 0
    (negval, posval) = (0,1)
    if model == 'p':
        negval = -1

    # modify y so that setosa --> -1 / 0 and everything else --> 1
    y = np.where(y == 'Iris-setosa', negval,posval)
        
    # extract sepal length and petal length (cols 0, 2)
    X = df.iloc[0:100, [0,2]].values

    # plot data to make sure it's linearly separable (otherwise perceptron won't work)

#   NOTE: THIS CODE IS NOT WORKING YET because X is an numpy Array, not a List.
#    series_neg = [val for is_good, val in zip(y,X) if is_good == negval]
#    series_pos = [val for is_good, val in zip(y,X) if is_good == posval]
#    plt.scatter(series_neg[,0], series_neg[,1], color='red', marker='o', label='Setosa')
#    plt.scatter(series_pos[:,0], series_pos[:,1], color='blue', marker = 'x', label='Virginica')

    plt.scatter(X[:50,0], X[:50,1], color='red', marker='o', label='Setosa')
    plt.scatter(X[50:100,0], X[50:100,1], color='blue', marker = 'x', label='Virginica')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.title('Iris data set')
    plt.legend()
    plt.show()

    return (X,y)

def runPerceptronModel():
    # Train perceptron
    X,y = loadDataset(model = 'p')
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X,y)
    plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('# Updated w parameters')
    plt.title('Convergence of Perceptron model')
    plt.show()
    
    # Plot decision boundary
    plot_decision_regions(X,y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.title('Iris classification (perceptron)')
    plt.legend()
    plt.show()
      
def runLRModel():
    norm = True
    X,y = loadDataset(model = 'lr')

    # Train logistic regression classifier
    #lrgd = LogisticRegressionGD(eta=0.01, n_iter = 200, random_state = 1, to_normalize = norm) # ML book notation
    lrgd = LogisticRegressionGD(alpha = 2.0, n_iter = 200, random_state = 1, to_normalize = norm) #A.Ng notation
    lrgd.fit(X,y)

    #Plot costs vs. # iterations to verify convergence
    plt.plot(range(1, len(lrgd.cost_) + 1),lrgd.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Gradient descent cost function')
    plt.title('Convergence of Logistic Regression model')
    plt.show()

    #Plot decision boundaries
    plot_decision_regions(X,y, classifier=lrgd)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.title('Iris classification (logistic regression); Used normalization? '+str(norm))
    plt.legend()
    plt.show()
    
    # Run a few test cases through
    print('Running a few test cases...')
    print(lrgd.predict(np.array([[3.,5.]])),'(should be 1)') # should be 1
    print(lrgd.predict(np.array([[7.,1.]])),'(should be 0)') # should be 0
    print(lrgd.predict(np.array([[5.,3.]])),'(should be 1)') # should be 1
    print(lrgd.predict(np.array([[7.,4.]])),'(should be 1)') # should be 1
    print(lrgd.predict(np.array([[6.,1.]])),'(should be 0)') # should be 0
    
   

#runPerceptronModel()
runLRModel()
