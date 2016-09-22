% COMP 6321 Machine Learning, Fall 2016
% Federico O'Reilly Regueiro - 40012304
% Assignment 1, due September 30

% Note: albeit non-standard, in order to create a single m file, the ' script' 
% part has been wrapped in a function that takes no parameters and returns 
% nothing...

function [] = A1  % Q 1a
    x = load('hw1x.dat');
    y = load('hw1y.dat');

    % plot x against y, use circles and avoid plotting connecting lines
    figure(1);
    plot(x, y, 'o');
    title(['Comp 6321, Fall 2016 - A1 - Federico O''Reilly Regueiro -', ... 
           ' data plot and regressions d = [1:3]']);
    ylabel('y');
    xlabel('x');

    % Q 1b
    x = [x, ones(length(x),1)];

    % w = inv(x'*x) * x'*y;
    w = (x' * x) \ x'*y;

    hold on;

    % just plot the lie between the en
    endpoints = [min(x), max(x)];

    plot(endpoints, (w(1).* endpoints + w(2)), 'r');

    jh_lin = trainingErr(x, w, y);

    % Q 1e - TODO is this a better fit? write in doc
    [w2, x_prime] = PolyRegress(x,y,2);
    plot(x(:,size(x,2)-1), x_prime*w2, 'g.')

    jh_quad = trainingErr(x_prime, w2, y);

    % Q 1f - TODO is this a better fit? write in doc
    [w3, x_prime] = PolyRegress(x,y,3);
    plot(x(:,size(x,2)-1), x_prime*w3, 'k+')

    jh_cub = trainingErr(x_prime, w3, y);

    legend('data points', ['linear regression, MSE=',num2str(jh_lin)], ...
          ['quadratic regression, MSE=', num2str(jh_quad)], ...
          ['cubic regression, MSE=', num2str(jh_cub)]);
    hold off;

    % Q 1h
    [d, train_error, test_error] = k_fold_cv (x, y, 5);
    figure(2);
    plot(test_error);
    hold on;
    plot(2:d+1,diff(test_error), 'r.');
    test_error
    diff(test_error)
    hold off;
    title(['Comp 6321, Fall 2016 - A1 - Federico O''Reilly Regueiro -', ... 
           ' data plot and regressions d = [1:3]']);
    ylabel('validation MSE');
    xlabel('d - order of polynomial regression');

    [wd, x_prime] = PolyRegress(x,y,d);
    figure(3);
    plot(x(:,1), y, 'o');
    hold on;
    plot(x(:, size(x,2)-1), x_prime*wd, 'r*');, 
    hold off;
end

% Q 1c
function jh = trainingErr(x, w, y)
  hx = x*w;
  % mean squared error? slides have the sum of squares but talk about mse...
  jh = sum((hx - y).^2)/(2*size(x,1));
end

% Q 1d
function [w, x_prime] = PolyRegress(x,y,d)
  if(d >= 1)
    x_prime = format_poly(x,d);
  else
    error('Error, d must be larger or equal to 1');
  end
  w = (x_prime'*x_prime) \ x_prime'*y;
end

% helper for PolyRegress
function x_prime = format_poly(x,d) % TODO continue here and propagate normalization
    x_prime = repmat(x(:,1), 1, d+1).^(repmat((d:-1:0), size(x,1), 1));
end

function [d, train_error, test_error] = k_fold_cv (x, y, k, normalize)
    % sure, we could solve for the general case, but...
    if mod(size(x,1), k)
        error('Matrix x must contain a multiple of k entries');
    end
    if nargin < 3
        normalize = false;
    end
    m = size(x,1);
    % x_shuffle = x(randperm(m),:);
    x_shuffle = 1:m;
    
    % matlab abonimation for a do-while loop
    test_err_decreasing = true;
    % first entry in the errors vector is the maximum possible fp num
    % just for the logic in the while loop (no do-while in matlab)
    % get rid of these two entries at the end
    test_error = realmax;
    train_error = realmax;
    d = 0;
    while (d<8)%(test_err_decreasing)
        % create a blank entry to accumulate errors each fold
        d = d+1; 
        test_error = [test_error; 0];
        train_error = [train_error; 0];
        m_k = m/k;
        for i = 1:k
           idx_btm = 1:(m_k*(i-1));
           idx_tst = m_k*(i-1)+1:(m_k*i);
           idx_top = (m_k*i)+1:m;
           x_train = [x(idx_btm,:);x(idx_top,:)];
           y_train = [y(idx_btm,:);y(idx_top,:)];
           x_test = format_poly(x(idx_tst, :), d);
           y_test = y(idx_tst, :);

           % training
           %w = (x_train'*x_train) \ x_train'*y_train;
           [w, x_train_prime] = PolyRegress(x_train,y_train,d);
           jh_train = trainingErr(x_train_prime, w, y_train);
           
           train_error(d+1) = train_error(d+1) + jh_train;
           % validation
           jh_test = trainingErr(x_test, w, y_test);
           test_error(d+1) = test_error(d+1) + jh_test;
        end
        % TODO should our test allow one isolated small increase?
        % TODO also, change condition above used now to 'hack' d
        test_err_decreasing = test_error(d+1) < test_error(d);  
    end
    % TODO change the format of returned errors so we can get a variance
    % remove first entry for errors vectors
    test_error(1,:) = [];
    train_error(1,:) = [];
    % divide errors between k
    test_error = test_error/k;
    train_error = train_error/k;
    % d already stopped decreasing, optimal d has been passed
    d = d-1;
end

  
