function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta




i=1;
while i<=m
P = sigmoid(theta(1)*X(i,1) + theta(2)*X(i,2) +theta(3)*X(i,3)  );
J = J + y(i)*log(P) + (1 - y(i))*log(1 -P);
i = i+1;
end
J = -1 .* J;
J = J ./ m;
% regularized cost 
J = J + (lambda/(2*m))*(theta(2)*theta(2) +theta(3)*theta(3)); 

i=1;
a=0;
while i<=m
P = sigmoid(theta(1)*X(i,1) + theta(2)*X(i,2) +theta(3)*X(i,3)  );
a= 	a + ((P -y(i))*X(i,1));
i = i+1;
end



i=1;
b=0;
while i<=m
P = sigmoid(theta(1)*X(i,1) + theta(2)*X(i,2) +theta(3)*X(i,3)  );
b =b+ ((P -y(i))*X(i,2));
i = i+1;
end


i=1;
c=0;
while i<=m
P = sigmoid(theta(1)*X(i,1) + theta(2)*X(i,2) +theta(3)*X(i,3)  );
c=	c+ ((P -y(i))*X(i,3));
i = i+1;
end


grad = [a;b;c];
grad = grad ./ m;
grad(2) =grad(2) +(lambda/m)*(theta(2));
grad(3) = grad(3) + (lambda/m)*(theta(3));

%regularized theta


% =============================================================

end
