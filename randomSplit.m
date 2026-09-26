%Train Test Split function
%Didn't have the helper function so made it myself
%takes the percent test size, features, labels, and random seed
function [trainData, testData, trainLabels, testLabels] = randomSplit(test_size, feats, labels, seed)
%Setup
if nargin < 4 || isempty(seed)
    error('A random seed is required.');
end
rng(seed);
if nargin < 1 || isempty(test_size)
    test_size = 0.2;
end

if test_size > 1
    test_size = test_size / 100;
end

cv = cvpartition(labels, 'HoldOut', test_size);
trainData = feats(training(cv), :);
testData = feats(test(cv), :);
trainLabels = labels(training(cv));
testLabels = labels(test(cv));
end
