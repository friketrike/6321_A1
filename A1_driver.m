% COMP 6321 Machine Learning, Fall 2016
% Federico O'Reilly Regueiro - 40012304
% Assignment 1, due September 30

%%%% Q 1a %%%%
x = load('hw1x.dat');
y = load('hw1y.dat');

% plot x against y, use circles and avoid plotting connecting lines
figh1 = figure(1);
set(figh1,'Units','normalized');
set(figh1,'Position',[0 0.2 0.8 0.7]);
plot(x, y, 'o');
title('data and regressions d = [1:3]');
ylabel('y / hw(x)');
xlabel('x');

%%%% Q 1b %%%%
x = [x, ones(length(x),1)];

% w = inv(x'*x) * x'*y;
w = (x' * x) \ x'*y;

hold on;

% just plot the lie between the en
endpoints = [min(x), max(x)];

plot(endpoints, (w(1).* endpoints + w(2)), 'r');

jh_lin = trainingErr(x, w, y);

%%%% Q 1e %%%% TODO is this a better fit? write in doc
[w2, x_prime] = PolyRegress(x,y,2);
plot(x(:,size(x,2)-1), x_prime*w2, 'g.')

jh_quad = trainingErr(x_prime, w2, y);

%%%% Q 1f %%%% TODO is this a better fit? write in doc
[w3, x_prime] = PolyRegress(x,y,3);
plot(x(:,size(x,2)-1), x_prime*w3, 'k+')

jh_cub = trainingErr(x_prime, w3, y);

legend('data points', ['linear, MSE=',num2str(jh_lin)], ...
  ['quadratic, MSE=', num2str(jh_quad)], ...
  ['cubic, MSE=', num2str(jh_cub)], 'location', 'southeast');
legend boxoff
hold off;

%print fig1.pdf

%%%% Q 1g - support %%%%
support_1g;

%%%% Q 1h %%%%
[d, train_error, test_error] = k_fold_cv (x, y, 5);
figh2 = figure(2);
set(figh2,'Units','normalized');
set(figh2,'Position',[0.1 0.15 0.8 0.7]);
subplot(1,2,1);
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
title('Data and regressions');
ylabel('validation MSE');
xlabel('d');
legend('mean of testing errors (mte)', ...
   'negative fd(mte)', ...
   'positive fd(mte)', 'location', 'southeast');
legend boxoff

[wd, x_poly] = PolyRegress(x,y,d);
jh_d = trainingErr(x_poly, wd, y);
subplot(1,2,2);
plot(x(:,1), y, 'o');
hold on;
plot(x(:, size(x,2)-1), x_poly*wd, 'r*'); 
hold off;
title('Data and polynomial regr. d obtained via 5-fold cv');
ylabel('y / hw(x)');
xlabel('x');
legend('data', [num2str(d), '-order regr., MSE=', ...
    num2str(jh_d)], 'location', 'southeast');
legend boxoff
    
%print fig2.pdf

% display to the console
mean_train_error = mean(train_error, 2)
mean_test_error = mean(test_error, 2)
    
%%%% Q 1i %%%%
[d, train_error_norm, test_error_norm] = k_fold_cv (x, y, 5, true);
figh3 = figure(3);
set(figh3,'Units','normalized');
set(figh3,'Position',[0.6 0.1 0.4 0.7]);
%%subplot(1,2,1);
plot(mean(test_error_norm, 2));
hold on;
% Plot the finite difference for the test errors
diff_abscissa = 2:size(test_error_norm,1);
test_err_norm_diff = diff(mean(test_error_norm,2));
% plot non-positive finite difference
idx = test_err_norm_diff < 0;
plot(diff_abscissa(idx),test_err_norm_diff(idx), 'g*' );
% plot positive finite difference
idx = test_err_norm_diff >= 0;
plot(diff_abscissa(idx),test_err_norm_diff(idx), 'rx' );
plot(1:d+2, zeros(d+2, 1), 'k-');
hold off;
title('Data and regressions, normalized');
ylabel('validation MSE');
xlabel('d');
legend('mean of testing errors (mte)', ...
   'negative fd(mte)', ...
   'positive fd(mte)', 'location', 'southeast');
legend boxoff

%print fig3.pdf
   
% display to the console
mean_train_error_norm = mean(train_error_norm, 2)
mean_test_error_norm = mean(test_error_norm, 2)

%%%% Q1 j - support %%%%
[wd_n, x_poly_n] = PolyRegress(x,y,d, true);
jh_d = trainingErr(x_poly_n, wd_n, y);
figure(4);
plot(x(:,1), y, 'o');
hold on;
plot(x(:,1)./max(x(:,1)), x_poly_n*wd_n, 'r*'); 
hold off;
title('Data and normalized polynomial regr.');
ylabel('y / hw(x)');
xlabel('x');
legend('data', [num2str(d), '-order regresion, MSE=', ...
    num2str(jh_d)], 'location', 'southeast');
legend boxoff

%print fig4.pdf
    
%%%% Q2 %%%%
Q2;



