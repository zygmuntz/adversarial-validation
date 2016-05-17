#!/usr/bin/env python

"train logistic regression and random forest, output predictions"

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF

#

lr_predictions_file = 'p_lr.csv'
rf_predictions_file = 'p_rf.csv'

#

train = pd.read_csv( 'data/train.csv' )
test = pd.read_csv( 'data/test.csv' )

x_train = train.drop( [ 'TARGET', 'ID' ], axis = 1 )
y_train = train.TARGET

x_test = test.drop( 'ID', axis = 1 )

#

print "LR..."

clf = LR()
clf.fit( x_train, y_train )

# predict

p_lr = clf.predict_proba( x_test )[:,1]
test['p_lr'] = p_lr

test.to_csv( lr_predictions_file, index = None, columns = [ 'ID', 'p_lr' ], header = [ 'ID', 'TARGET' ])

# private leaderboard: 0.614733

#

print "RF..."

clf = RF( n_estimators = 100, verbose = True, n_jobs = -1 )
clf.fit( x_train, y_train )

p_rf = clf.predict_proba( x_test )[:,1]
test['p_rf'] = p_rf

test.to_csv( rf_predictions_file, index = None, columns = [ 'ID', 'p_rf' ], header = [ 'ID', 'TARGET' ])

# private leaderboard: 0.743709
