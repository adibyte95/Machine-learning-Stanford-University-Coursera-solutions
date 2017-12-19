function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

i = 1;
while i<=m
J = J + (( (X(i,:) * theta) -y(i)) * ( (X(i,:) * theta) -y(i)))/(2*m) ;
i = i+1;
end;

i = 2;
while i<=size(theta)(1)
J = J + (lambda /(2*m))*(theta(i) * theta(i));
i = i +1;
end;

i = 1;
j = 1;
while i<=size(theta)(1)
	while j<=m
		if i == 1
		grad(i) = grad(i) + ( ( (X(j,:) * theta) -y(j) )*X(j,i) )/m ;
		end;
		if i~=1
		% m*m is there because the last term is multiplied m times more than it should so it is divided by m
		grad(i) = grad(i) + ((X(j,:) * theta) -y(j))*X(j,i)/m + (lambda/(m*m))*(theta(i));
		end;
	j = j +1;
	end;
	i = i+1;
	j = 1;
	end;
% =========================================================================
grad = grad(:);

end
