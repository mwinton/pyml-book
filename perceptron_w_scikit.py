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
#X_01 = X[(y==0) | (y==1)]
#y_01 = y[(y==0) | (y==1)]
#print('Class labels:', np.unique(y_01))
#X_train, X_test, y_train, y_test = tts(X_01,y_01,test_size=0.3, random_state=1, stratify=y_01)


# Split original data into train/test sets with equal proportions of each class
print('Class labels:', np.unique(y))
X_train, X_test, y_train, y_test = tts(X,y,test_size=0.3, random_state=1, stratify=y)


print('Label counts in y:', np.bincount(y))
print('Label counts in y_train:', np.bincount(y_train))
print('Label counts in y_test:', np.bincount(y_test))


# Apply feature scaling to standardize the train/test sets. (Both use mu/sigma from training)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Train perceptron
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

# Make prediction for test set
y_pred = ppn.predict(X_test_std)
print('Correctly classified samples: %d' % (y_test == y_pred).sum())
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy: %.2f' % ppn.score(X_test_std, y_test)) #equivalent to above


# Plot data and decision regions
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, 
                      y=y_combined, 
                      classifier=ppn, 
                      test_idx=range(y_train.shape[0],y_train.shape[0]+y_test.shape[0]))
plt.xlabel('petal length (standardized)')
plt.ylabel('petal width (standardized)')
plt.legend()
plt.show()



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
