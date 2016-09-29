[d, train_error_ord, test_error_ord] = k_fold_cv (x, y, 5, false, true, 9);
d
mean_train_error_ord = mean(train_error_ord, 2)
mean_test_error_ord = mean(test_error_ord, 2)