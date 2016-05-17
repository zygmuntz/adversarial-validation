#!/usr/bin/env python

"train and validate logistic regression and random forest"

import numpy as np
import pandas as pd

from sklearn import cross_validation as CV

from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF

from sklearn.metrics import roc_auc_score as AUC

#

train = pd.read_csv( 'data/train.csv' )

x = train.drop( [ 'TARGET', 'ID' ], axis = 1 )
y = train.TARGET

x_train, x_test, y_train, y_test = CV.train_test_split( x, y, test_size = 0.2 )

#

print "LR..."

clf = LR()
clf.fit( x_train, y_train )

# predict

p = clf.predict_proba( x_test )[:,1]
auc = AUC( y_test, p )
print "AUC: {:.2%}".format( auc )

# AUC: 58.30%

#

print "RF..."

clf = RF( n_estimators = 100, verbose = True, n_jobs = -1 )
clf.fit( x_train, y_train )

p = clf.predict_proba( x_test )[:,1]
auc = AUC( y_test, p )
print "AUC: {:.2%}".format( auc )

# AUC: 75.32%
