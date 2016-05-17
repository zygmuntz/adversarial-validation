#!/usr/bin/env python

"train (to distinguish between sets) and validate logistic regression and random forest"

import numpy as np
import pandas as pd

from sklearn import cross_validation as CV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF

from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score as accuracy

train = pd.read_csv( 'data/train.csv' )
test = pd.read_csv( 'data/test.csv' )

train['TARGET'] = 1
test['TARGET'] = 0

data = pd.concat(( train, test ))

# shuffle
data = data.iloc[ np.random.permutation(len( data )) ]
data.reset_index( drop = True, inplace = True )

x = data.drop( [ 'TARGET', 'ID' ], axis = 1 )
y = data.TARGET

train_examples = 100000
x_train = x[:train_examples]
x_test = x[train_examples:]
y_train = y[:train_examples]
y_test = y[train_examples:]

#

print "LR..."

clf = LR()
clf.fit( x_train, y_train )

# predict

p = clf.predict_proba( x_test )[:,1]
auc = AUC( y_test, p )
print "AUC: {:.2%}".format( auc )

# AUC: 50.06%

print "RF..."

clf = RF( n_estimators = 100, verbose = True, n_jobs = -1 )
clf.fit( x_train, y_train )

p = clf.predict_proba( x_test )[:,1]
auc = AUC( y_test, p )
print "AUC: {:.2%}".format( auc )

# AUC: 50.28%


"""
print "CV"

scores = CV.cross_val_score( LR(), x, y, scoring = 'roc_auc', cv = 2, verbose = 1 )	

print "mean AUC: {:.2%}, std: {:.2%} \n".format( scores.mean(), scores.std())

# mean AUC: 50.20%, std: 0.03%
"""
