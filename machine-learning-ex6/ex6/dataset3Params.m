function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0;
sigma =0;
cVec = [.01; .03; .1; .3; 1; 3; 10; 30];
sigmaVec = [.01; .03;.1; .3; 1; 3; 10; 30];
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% getting the model
min = 100000;
i =1;
j =1;
while i<=8
	while j <=8
		model = svmTrain(X,y,cVec(i), @(x1, x2) gaussianKernel(x1, x2, sigmaVec(j)));
		% now as the model is saved
		predictions = svmPredict(model, Xval);
		ans = mean(double(predictions ~= yval));
		% disp(ans);
		if ans < min
			C = cVec(i);
			sigma = sigmaVec(j);
			min = ans;
		end
	j = j +1;
	end;
j=1;
i = i +1;
end

% =========================================================================

end
