
function results = KNNClassifier(Xtrain, Xtest, ytrain, testLabels, classes, K)

if iscell(classes)
    classes = categorical(classes);
end

if (nargin < 6) || isempty(K)
    K = 5;
end

tic; %start timer when training starts
model = fitcknn(Xtrain, ytrain, NumNeighbors=K, Standardize=true);
predLabels = predict(model, Xtest);
ttTime = toc;

[TestVotes, TestCounts] = helperMajorityVote(predLabels, testLabels, classes);
testaccuracy = sum(eq(TestVotes, testLabels)) / numel(testLabels) * 100;
fprintf("The KNN model's test accuracy is %2.2f percent in %.3f seconds\n", testaccuracy, ttTime);
confMat = confusionmat(testLabels, TestVotes);

%set up the output
results.modelName = "K Nearest Neighbors";
results.model = model;
results.predLabels = predLabels;
results.recordVotes = TestVotes;
results.trueLabels = testLabels;
results.testAccuracy = testaccuracy;
results.confusionMatrix = confMat;
results.classes = classes;
results.time = ttTime;

end
