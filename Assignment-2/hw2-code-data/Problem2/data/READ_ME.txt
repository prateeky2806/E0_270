This folder contains the following files.

Regression_dataset.mat
----------------------
This file contains the dataset for the regression problem.

  train		      1000x2            
  train_y         1000x1           
  test			  1000x2
  test_y 		  1000x1


Regression_folds.mat
--------------------

The contents of this mat file are as follows:   

  fold_train        5x800x2           
  fold_train_y      5x800            
  fold_test         5x200x2         
  fold_test_y       5x200           



A fold is a partition of the training dataset (regression_dataset.mat in our case) into 2  parts. The first part is then used by the learning algorithm and the second part is used for testing the model learnt by the algorithm. In our case, we create 5 such folds, each containing its own fold_train and fold_test sets. These are created as follows:

The dataset is divided into 5 equal partitions. Now to create the test set for the i-th fold  (fold_test(i,:,:) ), we include only the data in the i-th partition and all other data except the i-th partition go into the training set of this fold(fold_train(i,:,:). The corresponding training labels are in fold_train_y(i,:) and test labels in fold_test_y(i,:)
