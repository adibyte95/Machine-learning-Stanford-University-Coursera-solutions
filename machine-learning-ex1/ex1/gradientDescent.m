function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

       i =1;
	   j = 0;
	   while i<=m
	     j = j + theta(1)*X(i,1) + theta(2)*X(i,2) - y(i) ;
		 i = i+1;
	   end
	   j = alpha * j;
	   j = j/m;
	   ans = j;
	   
	   i =1;
	   j = 0;
	   while i<=m
	     j = j + (theta(1)*X(i,1) + theta(2)*X(i,2) - y(i))*X(i,2); 
		 i = i+1;
	   end
	   j = alpha * j;
	   j = j/m;
	   
       theta(1) =theta(1) -ans;
       theta(2) =theta(2) - j;   




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end