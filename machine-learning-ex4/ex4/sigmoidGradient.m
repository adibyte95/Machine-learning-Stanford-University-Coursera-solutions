function g = sigmoidGradient (z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

info = size(z);
% rows and columns of the matrix
rows = info(1);
cols = info(2);

epsilon = .0001;
i=1;
j =1;
while i<=rows
while j<=cols
g(i,j)  = (sigmoid(z(i,j)+epsilon) - sigmoid(z(i,j)-epsilon))/(2*epsilon);
j = j+1;
end;
j =1;
i = i+1;
end;

% =============================================================




end
