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
    title('data plot and regressions d = [1:3]');
    ylabel('y / hw(x)');
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
    plot(mean(test_error, 2));
    hold on;
    % Plot the finite difference for the test errors
    diff_abscissa = 2:size(test_error,1);
    test_err_diff = diff(mean(test_error,2));
    % plot non-positive finite difference
    idx = test_err_diff < 0;
    plot(diff_abscissa(idx),test_err_diff(idx), 'g*' );
    % plot positive finite difference
    idx = test_err_diff >= 0;
    plot(diff_abscissa(idx),test_err_diff(idx), 'rx' );
    plot(1:d+2, zeros(d+2, 1), 'k-');
    hold off;
    title('Data plot and regressions');
    ylabel('validation MSE');
    xlabel('d - order of polynomial regression');
    legend('mean of testing errors (mte)', ...
           'negative fd(mte) ( ie. mte(d) - mte(d-1) < 0 )', ...
           'positive fd(mte)');
    [wd, x_prime] = PolyRegress(x,y,d);
    jh_d = trainingErr(x_prime, wd, y);
    figure(3);
    plot(x(:,1), y, 'o');
    hold on;
    plot(x(:, size(x,2)-1), x_prime*wd, 'r*'); 
    hold off;
    title(['Data plot and polynomial regresion of order ', num2str(d), ...
          ' obtained via 5-fold cross validation']);
    ylabel('y / hw(x)');
    xlabel('x');
    legend('data points', [num2str(d), '-order polynomial regresion, MSE=', ...
            num2str(jh_d)]);
            
            
            
            
    % Q 1i
    [d, train_error, test_error] = k_fold_cv (x, y, 5, true);
    figure(4);
    plot(mean(test_error, 2));
    hold on;
    % Plot the finite difference for the test errors
    diff_abscissa = 2:size(test_error,1);
    test_err_diff = diff(mean(test_error,2));
    % plot non-positive finite difference
    idx = test_err_diff < 0;
    plot(diff_abscissa(idx),test_err_diff(idx), 'g*' );
    % plot positive finite difference
    idx = test_err_diff >= 0;
    plot(diff_abscissa(idx),test_err_diff(idx), 'rx' );
    plot(1:d+2, zeros(d+2, 1), 'k-');
    hold off;
    title('Data plot and regressions, normalized');
    ylabel('validation MSE');
    xlabel('d - order of polynomial regression');
    legend('mean of testing errors (mte)', ...
           'negative fd(mte) ( ie. mte(d) - mte(d-1) < 0 )', ...
           'positive fd(mte)');
    [wd, x_prime] = PolyRegress(x,y,d, true);
    jh_d = trainingErr(x_prime, wd, y);
    figure(5);
    plot(x(:,1), y, 'o');
    hold on;
    plot(x(:, size(x,2)-1), x_prime*wd, 'r*'); 
    hold off;
    title(['Data plot and polynomial regresion of order ', num2str(d), ...
          ' obtained via 5-fold cross validation']);
    ylabel('y / hw(x)');
    xlabel('x');
    legend('data points', [num2str(d), '-order polynomial regresion, MSE=', ...
            num2str(jh_d)]);
end

% Q 1c
function jh = trainingErr(x, w, y)
  hx = x*w;
  % use mse, doesn't really matter but it's simpler to think of than total se
  jh = sum((hx - y).^2)/(2*size(x,1));
end

% Q 1d
function [w, x_prime] = PolyRegress(x, y, d, normalize)
  if nargin < 4
    normalize = false;
  end
  if(d >= 1)
    x_prime = format_poly(x, d, normalize);
  else
    error('Error, d must be larger or equal to 1');
  end
  w = (x_prime'*x_prime) \ x_prime'*y;
end

% helper for PolyRegress
function x_prime = format_poly(x,d, normalize) 
  if nargin < 3
    normalize = false;
  end  
  x_prime = repmat(x(:,1), 1, d+1).^(repmat((d:-1:0), size(x,1), 1));
  if normalize
  x_prime = x_prime / (diag(max(x_prime)));
  end
end

function [d, train_error, test_error] = k_fold_cv (x, y, k, normalize)
    % sure, we could solve for the general case, but...
    if mod(size(x,1), k)
        error('Matrix x must contain a multiple of k entries');
    end
    if nargin < 4
        normalize = false;
    end
    m = size(x,1);
    % x_shuffle = x(randperm(m),:);
    x_shuffle = 1:m;
    
    % matlab abonimation for a do-while loop
    test_err_decreasing = true;
    % first two entries in the errors are the maximum possible fp num
    % just for the logic in the while loop (no do-while in matlab)
    % get rid of these two entries at the end
    test_error = repmat(realmax, 2, k);
    train_error = repmat(realmax, 2, k);
    d = 0;
    while (test_err_decreasing)
        % create a blank entry to accumulate errors each fold
        d = d+1; 
        %test_error = [test_error; 0];
        %train_error = [train_error; 0];
        m_k = m/k;
        for i = 1:k
           idx_btm = 1:(m_k*(i-1));
           idx_tst = m_k*(i-1)+1:(m_k*i);
           idx_top = (m_k*i)+1:m;
           x_train = [x(idx_btm,:);x(idx_top,:)];
           y_train = [y(idx_btm,:);y(idx_top,:)];
           x_test = format_poly(x(idx_tst, :), d, normalize);
           y_test = y(idx_tst, :);

           % training
           %w = (x_train'*x_train) \ x_train'*y_train;
           [w, x_train_prime] = PolyRegress(x_train,y_train,d, normalize);
           jh_train = trainingErr(x_train_prime, w, y_train);
           
           train_error(d+2, k) = jh_train;
           % validation
           jh_test = trainingErr(x_test, w, y_test);
           test_error(d+2, k) = jh_test;
        end

        test_err_decreasing = (mean(test_error(d+2,:))...
             <= mean(test_error(d+1,:))) ...
                  || (mean(test_error(d+2,:)) < mean(test_error(d,:))); 
    end
    % remove first entry for errors vectors
    test_error(1:2,:) = [];
    train_error(1:2,:) = [];
    % divide errors between k
    test_error = test_error/k;
    train_error = train_error/k;
    % d already stopped decreasing, optimal d has been passed
    
    [dummy, d] = min(mean(test_error,2));
end

  
