function results = NaiveBayesClassifier(Xtrain, Xtest, ytrain, testLabels, classes)

if iscell(classes)
    classes = categorical(classes);
end

tic; %start timer when training starts
model = fitcnb(Xtrain, ytrain);
predLabels = predict(model, Xtest);
ttTime = toc;

[TestVotes, TestCounts] = helperMajorityVote(predLabels, testLabels, classes);
testaccuracy = sum(eq(TestVotes, testLabels)) / numel(testLabels) * 100;
fprintf("The Naive-Bayes model's test accuracy is %2.2f percent in %.3f seconds\n", testaccuracy, ttTime);
confMat = confusionmat(testLabels, TestVotes);

%set up the output
results.modelName = "Naive-Bayes";
results.model = model;
results.predLabels = predLabels;
results.recordVotes = TestVotes;
results.trueLabels = testLabels;
results.testAccuracy = testaccuracy;
results.confusionMatrix = confMat;
results.classes = classes;
results.time = ttTime;

end
