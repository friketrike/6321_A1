% COMP 6321 Machine Learning, Fall 2016
% Federico O'Reilly Regueiro - 40012304
% Assignment 1, due September 30

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

