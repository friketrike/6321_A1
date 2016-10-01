% COMP 6321 Machine Learning, Fall 2016
% Federico O'Reilly Regueiro - 40012304
% Assignment 1, due September 30

% calls the implementation of weighted linear regression with the assignment
% data and plots a data-set that might benefit from weighting

m = size(x,1);
[dummy, idx_max] = max(x(:,1));
u = ones(m,1);

% plot data again...
figure(8)
plot(x(:,1), y, 'o')
hold on;
legendStrs = {['data']};

% now add a weighting for the largest input value and leave all others at 1
% weight goes from .1 to 100
weights = [.1, 10, 100];
for i = 1:3
    u(idx_max) = weights(i);
    w = weighted_linear_regression(x, y, u);
    % plot what's going on
    plot(x(:,1), x*w, 'color', [1/i, i/3, 0.25]);
    legendStrs{i+1} = ['u[xmax] = ', num2str(u(idx_max))];
%    u(idx_max) = u(idx_max) * 1000;    
end
title('H varying according to scaling of error at x_max');
legend(legendStrs, 'location', 'southeast');
legend boxoff
hold off;
%print fig8.pdf

% now just add a few points for which uncertainty re data acquisition
% is really low, depending on the domain knowledge, it could seem like they're
% acquisition errors... those are good candidates for weighting, methinks.
figure(9)
bad_sensor = 5:.5:11;
bad_sensor = [bad_sensor(:), ones(length(bad_sensor), 1)];
x_bad = [x; bad_sensor];
y_bad = [y; -5*ones(length(bad_sensor),1)];
plot(x_bad(:,1), y_bad, 'o');
title('Same data with two points of higher uncertainty');
hold on;

% now perform linear regression with and without weighting
u = ones(size(x_bad, 1),1);
w = weighted_linear_regression(x_bad, y_bad, u);
m_bad = length(u);
u(m_bad-1:m_bad) = 0.01;
wu = weighted_linear_regression(x_bad, y_bad, u);

plot(x_bad(:,1), x_bad * w, 'xr');
plot(x_bad(:,1), x_bad * wu, '+g');
hold off;
legend('data', 'regr. w/equal weighting',...
       'lesser weighting to uncertain data');
legend boxoff;
%print fig9.pdf

