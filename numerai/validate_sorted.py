#!/usr/bin/env python

"Load sorted training set and validate on examples looking the most like test"

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression as LR

from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import log_loss

#

input_file = 'data/may/train_sorted.csv'
val_size = 5000

#

def train_and_evaluate( y_train, x_train, y_val, x_val ):

	lr = LR()
	lr.fit( x_train, y_train )

	p = lr.predict_proba( x_val )
	p_bin = lr.predict( x_val )

	acc = accuracy( y_val, p_bin )
	auc = AUC( y_val, p[:,1] )
	ll = log_loss( y_val, p[:,1] )
	
	return ( auc, acc, ll )
	
def transform_train_and_evaluate( transformer ):
	
	global x_train, x_val, y_train
	
	x_train_new = transformer.fit_transform( x_train )
	x_val_new = transformer.transform( x_val )
	
	return train_and_evaluate( y_train, x_train_new, y_val, x_val_new )
	
#

print "loading..."

data = pd.read_csv( input_file )

train = data.iloc[:-val_size]
val = data.iloc[-val_size:]

# print len( train ), len( val )

# 

y_train = train.target.values
y_val = val.target.values

x_train = train.drop( 'target', axis = 1 )
x_val = val.drop( 'target', axis = 1 )

# train, predict, evaluate

auc, acc, ll = train_and_evaluate( y_train, x_train, y_val, x_val )

print "No transformation"
print "AUC: {:.2%}, accuracy: {:.2%}, log loss: {:.2%} \n".format( auc, acc, ll )

# try different transformations for X

transformers = [ MaxAbsScaler(), MinMaxScaler(), RobustScaler(), StandardScaler(),  
	Normalizer( norm = 'l1' ), Normalizer( norm = 'l2' ), Normalizer( norm = 'max' ) ]

poly_scaled = Pipeline([ ( 'poly', PolynomialFeatures()), ( 'scaler', MinMaxScaler()) ])

transformers += [ poly_scaled ]

for transformer in transformers:

	print transformer
	auc, acc, ll = transform_train_and_evaluate( transformer )
	print "AUC: {:.2%}, accuracy: {:.2%}, log loss: {:.2%} \n".format( auc, acc, ll )

"""
No transformation
AUC: 52.54%, accuracy: 51.96%, log loss: 69.22%

MaxAbsScaler(copy=True)
AUC: 52.54%, accuracy: 51.98%, log loss: 69.22%

MinMaxScaler(copy=True, feature_range=(0, 1))
AUC: 52.54%, accuracy: 51.98%, log loss: 69.22%

RobustScaler(copy=True, with_centering=True, with_scaling=True)
AUC: 52.54%, accuracy: 52.04%, log loss: 69.22%

StandardScaler(copy=True, with_mean=True, with_std=True)
AUC: 52.53%, accuracy: 52.04%, log loss: 69.22%

Normalizer(copy=True, norm='l1')
AUC: 52.30%, accuracy: 52.46%, log loss: 69.23%

Normalizer(copy=True, norm='l2')
AUC: 52.35%, accuracy: 51.08%, log loss: 69.24%

Normalizer(copy=True, norm='max')
AUC: 52.37%, accuracy: 52.20%, log loss: 69.24%

Pipeline(steps=[('poly', PolynomialFeatures(degree=2, include_bias=True, interaction_only=
False)), ('scaler', MinMaxScaler(copy=True, feature_range=(0, 1)))])
AUC: 52.57%, accuracy: 51.76%, log loss: 69.58%
"""