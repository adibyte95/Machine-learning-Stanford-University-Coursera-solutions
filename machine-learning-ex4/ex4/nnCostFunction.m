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
X = [ones(m,1) X];

% part 1:

i =1;

while i<=m
% for the first layer
hidden_layer = sigmoid(Theta1* transpose(X(i,:)));
hidden_layer = [1; hidden_layer];

output_layer = sigmoid(Theta2 * hidden_layer); 

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

J = J -  transpose(new_y)* log(output_layer) - transpose(1 - new_y) * (log (1 -output_layer)) ; 
i = i+1;
end;

J = J/m;

% ------------- part 2 --------------------------------------------------------------------------------

% adding ones to the hidden layer

i=1;
while i<=m
hidden_layer = sigmoid(Theta1* transpose(X(i,:)));
hidden_layer = [1; hidden_layer];

output_layer = sigmoid(Theta2 * hidden_layer); 

yVec=[0;0;0;0];

if(y(i) == 1)
yVec = [1;0;0;0];
end;

if(y(i) == 2)
yVec = [0;1;0;0];
end;

if(y(i) == 3)
yVec = [0;0;1;0];
end;

if(y(i) == 4)
yVec = [0;0;0;1];
end;

delta3 = output_layer - yVec;
delta2 = (transpose(Theta2)* delta3) .*(hidden_layer .* (1 -hidden_layer));

Theta2_grad = Theta2_grad + delta3* transpose(hidden_layer);
Theta1_grad = Theta1_grad + delta2(2:end)*X(i,:);
i = i+1;
end;

Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;

%----------- PartTheta1_grad = (lambda/m)*(Theta1(:,2:end));
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



Theta1_new = Theta1;
Theta1_new(1,1) =0;
Theta1_new(2,1) =0;
Theta1_new(3,1) =0;
Theta1_new(4,1) = 0;

Theta2_new = Theta2;
Theta2_new(1,1) =0;
Theta2_new(2,1) = 0;
Theta2_new(3,1) =0;
Theta2_new(4,1) = 0;


Theta1_grad =Theta1_grad+ (lambda/m)*(Theta1_new);
Theta2_grad = Theta2_grad+ (lambda/m)*(Theta2_new);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
