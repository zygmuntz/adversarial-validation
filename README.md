# Adversarial validation

The `santander` dir holds the scripts for the Santander competition:

	distinguish_train_test.py - try to distinguish train/test set examples
	validate.py - get validation AUC scores for logistic regression and random forest
	predict.py - output test predictions from logistic regression and random forest

Similarly, the 'numerai' dir contains the Numerai scripts:

	distinguish_train_test.py - try to distinguish train/test set examples
	sort_train.py - sort training examples by their similarity to test examples
	validate_sorted.py - get validation scores using for most test-like examples
	predict.py - output test predictions
	
