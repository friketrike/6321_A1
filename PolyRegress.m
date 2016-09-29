% COMP 6321 Machine Learning, Fall 2016
% Federico O'Reilly Regueiro - 40012304
% Assignment 1, due September 30

%%%% Q 1d %%%%
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
