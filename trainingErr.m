% COMP 6321 Machine Learning, Fall 2016
% Federico O'Reilly Regueiro - 40012304
% Assignment 1, due September 30

%%%% Q 1c %%%%
function jh = trainingErr(x, w, y)
  hx = x*w;
  % use mse, doesn't really matter but it's simpler to think of than total se
  jh = sum((hx - y).^2)/(2*size(x,1));
end
