
function [] = test_folds()
  x = [.86;.09;-.85;.87;-.44;-.43;-1.1;.4;-.96;.17];
  x = [x, ones(length(x),1)];
  y = [2.49;.83;-.25;3.10;.87;.02;-.12;1.81;-.83;.43];

  [d, train_error, test_error] = k_fold_cv (x, y, 10);
  mean(train_error, 2);
  mean(test_error, 2);
end

% Q 1c
function jh = trainingErr(x, w, y)
  hx = x*w;
  % mean squared error? slides have the sum of squares but talk about mse...
  jh = sum((hx - y).^2)/(2.0*size(x,1));
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
    while (test_err_decreasing)
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
        test_err_decreasing = test_error(d+1) < test_error(d);  
    end
    % remove first entry for errors vectors
    test_error(1,:) = [];
    train_error(1,:) = [];
    % divide errors between k
    test_error = test_error/k;
    train_error = train_error/k;
    % d already stopped decreasing, optimal d has been passed
    d = d-1;
end