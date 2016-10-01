% COMP 6321 Machine Learning, Fall 2016
% Federico O'Reilly Regueiro - 40012304
% Assignment 1, due September 30

% calls the same k-fold validation process with folds taken from an ascending
% ordering of y in the data.

[d, train_error_ord, test_error_ord] = k_fold_cv (x, y, 5, false, true);
disp(['The chosen order with 5-fold validation performed on folds chosen ',...
      'with y ordered ascendingly is:', num2str(d)]);
      
% display the training and testing errors
mean_train_error_ord = mean(train_error_ord, 2)
mean_test_error_ord = mean(test_error_ord, 2)