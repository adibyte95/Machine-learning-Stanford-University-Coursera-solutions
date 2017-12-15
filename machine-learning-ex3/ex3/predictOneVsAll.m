function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

i=1;
while i<=m;
p1 = all_theta(1,1)*X(i,1) +all_theta(1,2)*X(i,2)  + all_theta(1,3)*X(i,3) ;
p2 = all_theta(2,1)*X(i,1) +all_theta(2,2)*X(i,2)  + all_theta(2,3)*X(i,3) ;
p3 = all_theta(3,1)*X(i,1) +all_theta(3,2)*X(i,2)  + all_theta(3,3)*X(i,3) ;
p4 = all_theta(4,1)*X(i,1) +all_theta(4,2)*X(i,2)  + all_theta(4,3)*X(i,3) ;
t = [p1;p2;p3;p4];
maximum = max(t);
if(p1 == maximum)
p(i) = 1;
end;
if(p2 == maximum)
p(i) = 2;
end;
if(p3 == maximum)
p(i) = 3;
end;
if(p4 == maximum)
p(i) = 4;
end; 
i = i +1;
end;


% =========================================================================


end
