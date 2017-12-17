function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% -------------------------------------------------------------


% part 1:

i =1;
while i<=m

% for the first layer
fsum1 =sigmoid(1*Theta1(1,1) + X(i,1)*Theta1(1,2) + X(i,2)*Theta1(1,3));
fsum2 =sigmoid(1*Theta1(2,1) + X(i,1)*Theta1(2,2) + X(i,2)*Theta1(2,3));
fsum3 =sigmoid(1*Theta1(3,1) + X(i,1)*Theta1(3,2) + X(i,2)*Theta1(3,3));
fsum4 = sigmoid(1*Theta1(4,1) + X(i,1)*Theta1(4,2) + X(i,2)*Theta1(4,3));

% for the second layer
ssum1 = sigmoid(1*Theta2(1,1) + fsum1*Theta2(1,2) + fsum2*Theta2(1,3) +fsum3*Theta2(1,4) + fsum4 *Theta2(1,5));
ssum2 = sigmoid(1*Theta2(2,1) + fsum1*Theta2(2,2) + fsum2*Theta2(2,3) +fsum3*Theta2(2,4) + fsum4 *Theta2(2,5));
ssum3 = sigmoid(1*Theta2(3,1) + fsum1*Theta2(3,2) + fsum2*Theta2(3,3) +fsum3*Theta2(3,4) + fsum4 *Theta2(3,5));
ssum4 = sigmoid(1*Theta2(4,1) + fsum1*Theta2(4,2) + fsum2*Theta2(4,3) +fsum3*Theta2(4,4) + fsum4 *Theta2(4,5));

if(y(i) ==1 )
new_y = [1;0;0;0];
end;

if(y(i) ==2 )
new_y = [0;1;0;0];
end;

if(y(i) ==3 )
new_y = [0;0;1;0];
end;

if(y(i) ==4 )
new_y = [0;0;0;1];
end;

J = J + new_y(1)*log(ssum1) +(1-new_y(1))*log(1-ssum1) + new_y(2)*log(ssum2) +(1-new_y(2))*log(1-ssum2) ;
J = J + new_y(3)*log(ssum3) +(1-new_y(3))*log(1-ssum3)  +new_y(4)*log(ssum4) +(1-new_y(4))*log(1-ssum4) 
i = i+1;
end;

J = -J/m;

% ------------- part 2 --------------------------------------------------------------------------------

%delta2 is the error for the final layer
% delta1 is the error for the hidden layer 

i=1;
while i<=m
fsum1 =sigmoid(1*Theta1(1,1) + X(i,1)*Theta1(1,2) + X(i,2)*Theta1(1,3));
fsum2 =sigmoid(1*Theta1(2,1) + X(i,1)*Theta1(2,2) + X(i,2)*Theta1(2,3));
fsum3 =sigmoid(1*Theta1(3,1) + X(i,1)*Theta1(3,2) + X(i,2)*Theta1(3,3));
fsum4 = sigmoid(1*Theta1(4,1) + X(i,1)*Theta1(4,2) + X(i,2)*Theta1(4,3));

% for the second layer
ssum1 = sigmoid(1*Theta2(1,1) + fsum1*Theta2(1,2) + fsum2*Theta2(1,3) +fsum3*Theta2(1,4) + fsum4 *Theta2(1,5));
ssum2 = sigmoid(1*Theta2(2,1) + fsum1*Theta2(2,2) + fsum2*Theta2(2,3) +fsum3*Theta2(2,4) + fsum4 *Theta2(2,5));
ssum3 = sigmoid(1*Theta2(3,1) + fsum1*Theta2(3,2) + fsum2*Theta2(3,3) +fsum3*Theta2(3,4) + fsum4 *Theta2(3,5));
ssum4 = sigmoid(1*Theta2(4,1) + fsum1*Theta2(4,2) + fsum2*Theta2(4,3) +fsum3*Theta2(4,4) + fsum4 *Theta2(4,5));


if(y(i) ==1 )
delta2_1 =ssum1 - 1;
delta2_2 =ssum2 - 0;
delta2_3 =ssum3 - 0;
delta2_4 =ssum4 - 0;
end;

if(y(i) ==2 )
delta2_1 =ssum1 - 0;
delta2_2 =ssum2 - 1;
delta2_3 =ssum3 - 0;
delta2_4 =ssum4 - 0;
end;

if(y(i) ==3 )
delta2_1 =ssum1 - 0;
delta2_2 =ssum2 - 0;
delta2_3 =ssum3 - 1;
delta2_4 =ssum4 - 0;
end;

if(y(i) ==4 )
delta2_1 =ssum1 - 0;
delta2_2 =ssum2 - 0;
delta2_3 =ssum3 - 0;
delta2_4 =ssum4 - 1;
end;


new_x = [ones(m,1) X];
delta2_new = [delta2_1;delta2_2;delta2_3;delta2_4];

first_layer = [1;fsum1;fsum2;fsum3;fsum4];

delta1_new = transpose(Theta2) *delta2_new .* (first_layer .*(1-first_layer)) ;

Theta2_grad = Theta2_grad + delta2_new *transpose(first_layer);
Theta1_grad = Theta1_grad + delta1_new(2:end) *new_x(i,:);

i= i+1;
end;

Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;

%----------- Part 3 -----------------------------------------------------------------
i =1;
j =2;
temp1 =0;
while i<=4
	while j<=3
	temp1= temp1 + Theta1(i,j)*Theta1(i,j);
	j = j+1;
	end;
i = i +1;
j=2;
end;


i=1;
j=2;
temp2 = 0;
while i<=4
	while j<=5
	temp2 = temp2 + (Theta2(i,j)*Theta2(i,j));
	j = j+1;
	end;
i = i +1;
j=2;
end;

J = J +(lambda/(2*m))*(temp1 + temp2);

i =1;
j =1;
while i<=4
	while j<=3
	if j == 1
	Theta1_grad(i,j) = Theta1_grad(i,j) ;
	end; 
	if j~= 1
	Theta1_grad(i,j) = Theta1_grad(i,j) + (lambda/m) *Theta1(i,j);
	end;
	j = j+1;
	end;
i = i+1;
j=1;
end;



i =1;
j =1;
while i<=4
	while j<=5
	if j == 1
	Theta2_grad(i,j) = Theta2_grad(i,j) ;
	end; 
	if j~= 1
	Theta2_grad(i,j) = Theta2_grad(i,j) + (lambda/m) *Theta2(i,j);
	end;
	j = j+1;
	end;
i = i+1;
j=1;
end;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
