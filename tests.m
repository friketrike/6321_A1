% random comment
function [] = tests()
    x = [1,2;3,4;5,6;7,8;9,10;11,12];
    x_3 = partition_matrix(x,3)
    size(x_3,3)
    
    for i = 1:6
       botm = 1:(5*(i-1))
       testt = 5*(i-1)+1:(5*i)
       topp = (5*i)+1:30     
    end
end

function x_k = partition_matrix(x, k)
   m = size(x,1);
   n = size(x,2);
   if mod(m, k)
       error('Matrix x must contain a multiple of k entries');
   else
       x_k = reshape(x(randperm(m),:),m/k, n, k);
   end
end
