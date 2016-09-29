% COMP 6321 Machine Learning, Fall 2016
% Federico O'Reilly Regueiro - 40012304
% Assignment 1, due September 30

% takes a matrix of the k hypotheses (as columns) given by k-fold validation 
% for a certain order d and plots them against the the data.
function [] = plot_h_space(x, y, d, h_space)
k = size(h_space, 2);
x = x(:,1);
x_poly = format_poly(x, d);
figure(3+d);
plot(x, y, '.');
hold on;
  for i = 1:k
      plot(x, x_poly*h_space(:,i), '.', 'color', [i/k, 0.5, 0.5])
  end
hold off;
end