% COMP 6321 Machine Learning, Fall 2016
% Federico O'Reilly Regueiro - 40012304
% Assignment 1, due September 30

function w = weighted_linear_regression(x, y, u)
    u = diag(u);
    w = (x' * u * x) \ x' * u * y;
end