% COMP 6321 Machine Learning, Fall 2016
% Federico O'Reilly Regueiro - 40012304
% Assignment 1, due September 30

%%%% Q1 h and Q1 i %%%%
function [d, train_error, test_error] = k_fold_cv (x, y, k, normalize, order_y)
    % sure, we could solve for the general case, but...
    if mod(size(x,1), k)
        error('Matrix x must contain a multiple of k entries');
    end
    if nargin < 5
        order_y = false;
    end
    if nargin < 4
        normalize = false;
    end
    m = size(x,1);
    % for question 1 g
    if order_y
        [dummy, idx_shuffle] = sort(y);
    else 
        %idx_shuffle = randperm(m);
        idx_shuffle  = 1:m;
    end
    
    % matlab abonimation for a do-while loop
    test_err_decreasing = true;

    % first two entries in the errors are the maximum possible fp num
    % just for the logic in the while loop (no do-while in matlab)
    % get rid of these two entries at the end
    test_error = repmat(realmax, 2, k);
    train_error = repmat(realmax, 2, k);

    d = 0;

    while (test_err_decreasing)

        d = d+1; 

        m_k = m/k;
        
        h_space = [];

        for i = 1:k
           idx_btm = idx_shuffle(1:(m_k*(i-1)));
           idx_tst = idx_shuffle(m_k*(i-1)+1:(m_k*i));
           idx_top = idx_shuffle((m_k*i)+1:m);
           x_train = [x(idx_btm,:);x(idx_top,:)];
           y_train = [y(idx_btm,:);y(idx_top,:)];
           x_test = x(idx_tst, :);
           y_test = y(idx_tst, :);

           % training
           [w, x_train_poly] = PolyRegress(x_train,y_train,d, normalize);
           jh_train = trainingErr(x_train_poly, w, y_train);           
           train_error(d+2, i) = jh_train;

           % validation
           x_test_poly = format_poly(x_test, d, normalize);
           jh_test = trainingErr(x_test_poly, w, y_test);
           test_error(d+2, i) = jh_test;
           
           % store w for plotting the hypotheses; support for Q1g
           h_space = [h_space, w];
        end
        
        if order_y
            plot_h_space(x, y, d, h_space);
        end
        
        test_err_decreasing = ...
              (mean(test_error(d+2,:)) <= mean(test_error(d+1,:))) ...
           || (mean(test_error(d+2,:)) < mean(test_error(d,:)));       
        
        % test_err_decreasing = d < 32;
    end

    % remove first two entries for errors vectors
    test_error(1:2,:) = [];
    train_error(1:2,:) = [];104

    % d already stopped decreasing, optimal d has been passed 
    [dummy, d] = min(mean(test_error,2));
end
