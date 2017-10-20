#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 21:24:25 2017

@author: mike
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt
import numpy as np


iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

# Subset data set to only classes 0,1
X_01 = X[(y==0) | (y==1)]
y_01 = y[(y==0) | (y==1)]
print('Class labels:', np.unique(y_01))
X_train, X_test, y_train, y_test = tts(X_01,y_01,test_size=0.3, random_state=1, stratify=y_01)


# Split original data into train/test sets with equal proportions of each class
#print('Class labels:', np.unique(y))
#X_train, X_test, y_train, y_test = tts(X,y,test_size=0.3, random_state=1, stratify=y)


print('Label counts in y:', np.bincount(y))
print('Label counts in y_train:', np.bincount(y_train))
print('Label counts in y_test:', np.bincount(y_test))


# Apply feature scaling to standardize the train/test sets. (Both use mu/sigma from training)
print('Applying standardization to data set')
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Train Logistic Regression classified
#lrgd = LogisticRegressionGD(n_iter=40, eta=0.1, random_state=1) # Py ML book
lrgd = LogisticRegressionGD(n_iter=100, alpha=1.0, random_state=1) # Andrew Ng
lrgd.fit(X_train_std, y_train)


# Make prediction for test set
y_pred = lrgd.predict(X_test_std)
print('Correctly classified samples: %d' % (y_test == y_pred).sum())
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
#print('Accuracy: %.2f' % lrgd.score(X_test_std, y_test)) #equivalent to above


#Plot costs vs. # iterations to verify convergence
plt.plot(range(1, len(lrgd.cost_) + 1),lrgd.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Gradient descent cost function')
plt.title('Convergence of Logistic Regression model')
plt.show()


# Plot data and decision regions
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, 
                      y=y_combined, 
                      classifier=lrgd, 
                      test_idx=range(y_train.shape[0],y_train.shape[0]+y_test.shape[0]))
plt.title('Iris Classification by logistic regression')
plt.xlabel('petal length (standardized)')
plt.ylabel('petal width (standardized)')
plt.legend()
plt.show()

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
    
    def __init__(self, alpha = 2.0, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta #Py ML book
        self.alpha = alpha # Andrew Ng
        self.n_iter = n_iter
        self.random_state = random_state


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
        
    def net_input(self, X):
        """Calculate net input.  This term is z in Andrew Ng's class """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def sigmoid (self, z):
        '''Calculate sigmoid function fron net_input (z)'''     
        # z values > 250 (or < -250) are clipped at 250
        return 1.0 / (1.0 + np.exp(-np.clip(z,-250,250)))
        
    def cost_function (self,y,output):
        '''Calculate the logistic regression cost function'''
        m = len(y)
        cost = (1/m) * (-y.dot(np.log(output)) - ((1.-y).dot(np.log(1.-output))))
        return cost
                    
    def predict(self, X):
        """Return class label based on sigmoid function"""
        
        #Due to shape of sigmoid function, these two options are equivalent
        #return np.where(self.net_input(X) >= 0, 1, 0)
        return np.where(self.sigmoid(self.net_input(X)) >= 0.5, 1, 0)
 



def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    '''
    Helper function to plot the decision regions.
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
 
    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx,:], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:,1],
                    alpha=1.0, 
                    c='',
                    linewidth = 1,
                    marker='o',
                    s=100,
                    label='test set', 
                    edgecolor='black')
