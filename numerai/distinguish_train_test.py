#!/usr/bin/env python

"cross-validate a random forest to distinguish between train and test"

import pandas as pd

from sklearn import cross_validation as CV
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score as accuracy

#

train_file = 'data/april/train.csv'
test_file = 'data/april/test.csv'

print "loading..."

train = pd.read_csv( train_file )
train.drop( 'target', axis = 1 , inplace = True )

test = pd.read_csv( test_file )
test.drop( 't_id', axis = 1 , inplace = True )

train['is_test'] = 0
test['is_test'] = 1

orig_train = train.copy()

train = pd.concat(( orig_train, test ))
train['target'] = train.is_test
train.drop( 'is_test', axis = 1, inplace = True )

#

print "cross-validating logistic regression..."

clf = LR()

scores = CV.cross_val_score( clf, train.drop( 'target', axis = 1 ), train.target, 
	scoring = 'roc_auc', cv = 5, verbose = 1 )
	
print "mean AUC: {:.2%}, std: {:.2%} \n".format( scores.mean(), scores.std())	

# mean AUC: 52.66%, std: 0.33%



print "cross-validating random forest..."

n_estimators = 100
clf = RF( n_estimators = n_estimators, n_jobs = -1, verbose = True )

scores = CV.cross_val_score( clf, train.drop( 'target', axis = 1 ), train.target, 
	scoring = 'roc_auc', cv = 5, verbose = 1 )
	
print "mean AUC: {:.2%}, std: {:.2%} \n".format( scores.mean(), scores.std())	

# mean AUC: 87.14%, std: 0.49%
