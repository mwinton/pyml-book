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
    
    def __init__(self, eta=0.05, n_iter=100, random_state=1, to_normalize=False):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.to_normalize = to_normalize

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
        
#       NORMALIZATION ISN'T WORKING YET 
        if self.to_normalize:
            X = self.normalize(X)
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        #NOTE: Need to switch to gradient descent, rather than blindly doing n_iter iterations
        #NOTE: Need to switch from eta to 1/m 
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.sigmoid(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            self.cost_.append(self.cost_function(y, output))
            
        return self

    def net_input(self, X):
        """Calculate net input.  This term is z in Andrew Ng's class """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def sigmoid (self, z):
        '''Calculate sigmoid function fron net_input (z)'''     
        # z values > 250 (or < -250) are clipped at 250
        return 1. / (1. + np.exp(-np.clip(z,-250,250)))
        
    def cost_function (self,y,output):
        '''Calculate the logistic regression cost function'''
        # NOTE: need to include the 1/m term
        # NOTE: need to update to return both J and Gradient
        return (-y.dot(np.log(output)) - ((1.-y).dot(np.log(1.-output))))
        
    def gradient_descent (self, X, y, w):
        '''Performs gradient descent to optimize'''
        #NOTE: need to implement
        pass
    
    def normalize (self, X):
        '''Normalizes feature vectors'''
        
        X_norm = np.copy(X)
        num_cols = X_norm.shape[1]
        self.X_means = [] # list will have one entry for each feature column of X
        self.X_stds = [] # list will have one entry for each feature column of X
        
        for i in range(0,num_cols):
            X_norm[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()
            # To normalize by range (instead of sigma) use the following line:
            #X_norm[:,i] = (X[:,i] - X[:,i].mean()) / (X[:,i].max()-X[:,i].min())

            #Store normalization factors to restore predicted values later
            self.X_means.append(X[:,i].mean())
            self.X_stds.append(X[:,i].std())
            # To normalize by range (instead of sigma) use the following line:
            #self.X_stds.append((X[:,i].max()-X[:,i].min()))
        print('During normalization, calculated mean=',self.X_means,'std=',self.X_stds)
        return X_norm
        
    def predict(self, X):
        """Return class label based on sigmoid function"""
        
        if self.to_normalize:
            X_norm = np.copy(X)
            for i in range(0,X_norm.shape[1]):
                # During prediction, need to use the saved means/stds, rather than recalculate
                X_norm[:,i] = (X[:,i] - self.X_means[i]) / self.X_stds[i]
            return np.where(self.net_input(X_norm) >= 0, 1, 0)
        else:
            #Due to shape of sigmoid function, these two options are equivalent
            return np.where(self.net_input(X) >= 0, 1, 0)
#           return np.where(self.sigmoid(self.net_input(X)) >= 0.5, 1, 0)




def plot_decision_regions(X, y, classifier, resolution=0.02):

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
 
def prepareDataset(model = 'lr'):
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
    negval = 0
    posval = 1
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
    X,y = prepareDataset(model = 'p')
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
    # Train logistic regression classifier
    norm = True
    X,y = prepareDataset(model = 'lr')
    lrgd = LogisticRegressionGD(eta=0.05, n_iter = 10000, random_state = 1, to_normalize = norm)
    lrgd.fit(X,y)

    plot_decision_regions(X,y, classifier=lrgd)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.title('Iris classification (logistic regression); Normalized= '+str(norm))
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
