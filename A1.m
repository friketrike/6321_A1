% COMP 6321 Machine Learning, Fall 2016
% Federico O'Reilly Regueiro - 40012304
% Assignment 1, due September 30
function [] = A  % Q 1a
    x = load('hw1x.dat');
    y = load('hw1y.dat');

    % plot x against y, use circles and avoid plotting connecting lines
    plot(x, y, 'o');
    title('Comp 6321, Fall 2016 - Assignment 1 - Federico O''Reilly Regueiro');
    ylabel('y');
    xlabel('x');

    % Q 1b
    x = [x, ones(length(x),1)];

    w = inv(x'*x) * x'*y;

    hold on;

    endpoints = [min(x), max(x)];

    plot(endpoints, (w(1)+endpoints.*w(2)), 'r');

    trainingErr(x, w, y, 'linear hypothesis');

    % Q 1e - TODO is this a better fit? write in doc
    [w2, x_prime] = PolyRegress(x,y,2);
    plot(x(:,size(x,2)-1), x_prime*w2, 'g.')

    trainingErr(x_prime, w2, y, 'quadratic hypothesis');

    % Q 1f - TODO is this a better fit? write in doc
    [w3, x_prime] = PolyRegress(x,y,3);
    plot(x(:,size(x,2)-1), x_prime*w3, 'k+')

    trainingErr(x_prime, w3, y, 'cubic hypothesis');

    legend('data points', 'linear regression', 'quadratic regression', 'cubic regression');
    hold off;
end

% Q 1c
function jh = trainingErr(x, w, y, h_class)
  hx = x*w;
  % mean squared error? slides have the sum of squares but talk about mse...
  jh = sum((hx - y).^2)/(2*size(x,1));
  fprintf('The training error for the %s is %d\n', h_class, jh);
end

% Q 1d
function [w, x_prime] = PolyRegress(x,y,d)
  if(d > 1)
    x_prime = [repmat(x(:,1), 1, d-1).^(repmat((2:d), size(x,1), 1)), x];
  else
    error('Error, d must be larger than 1');
  end
  w = inv(x_prime'*x_prime) * x_prime'*y;
end
  
