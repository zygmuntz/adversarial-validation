#!/usr/bin/env python

"Load data, train, output predictions"

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression as LR

train_file = 'data/may/train.csv'
test_file = 'data/may/test.csv'
lr_output_file = 'data/predictions_lr.csv'
poly_output_file = 'data/predictions_poly.csv'

#

print "loading..."

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

x_train = train.drop( 'target', axis = 1 )
y_train = train.target.values

x_test = test.drop( 't_id', axis = 1 )

print "training..."

lr = LR()
print lr
lr.fit( x_train, y_train )

poly = make_pipeline( PolynomialFeatures(), LR()) 
print poly
poly.fit( x_train, y_train )

print "predicting..."

p_lr = lr.predict_proba( x_test )
test['p_lr'] = p_lr[:,1]

p_poly = poly.predict_proba( x_test )
test['p_poly'] = p_poly[:,1]

print "saving..."

test.to_csv( lr_output_file, columns = ( 't_id', 'p_lr' ), header = ( 't_id', 'probability' ), index = None )
test.to_csv( poly_output_file, columns = ( 't_id', 'p_poly' ), header = ( 't_id', 'probability' ), index = None )

# LR:	0.69101
# Poly:	0.69229
	