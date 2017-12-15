function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% initialize the first layer

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
t = [ssum1;ssum2;ssum3;ssum4];
maximum = max(t);
if(ssum1 == maximum)
p(i) = 1;
end;
if(ssum2 == maximum)
p(i) = 2;
end;
if(ssum3 == maximum)
p(i) = 3;
end;
if(ssum4 == maximum)
p(i) = 4;
end; 

i = i+1;
end;
disp(p);
% =========================================================================

end
