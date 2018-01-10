function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

%%  calculation of mean
i =1;
j = 1;
while i<=n
	mean_cal = 0;
	while j<=m
		mean_cal = mean_cal + X(j,i)/m; 
	j = j +1;
	end;
mu(i) = mean_cal;
j=1;
i = i+1;
end;

%% calculation of standered deviation
i =1;
j = 1;
while i<=n
	var_cal = 0;
	while j<=m
		var_cal = var_cal + ((X(j,i) - mu(i))*(X(j,i) - mu(i)))/m; 
	j = j +1;
	end;
sigma2(i) = var_cal;
j=1;
i = i+1;
end;


% =============================================================


end
