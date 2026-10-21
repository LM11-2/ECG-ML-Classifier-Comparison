function results = NaiveBayesClassifier(Xtrain, Xtest, ytrain, testLabels, classes)

if iscell(classes)
    classes = categorical(classes);
end

tic; %start timer when training starts
model = fitcnb(Xtrain, ytrain);
predLabels = predict(model, Xtest);
ttTime = toc;

[TestVotes, TestCounts] = helperMajorityVote(predLabels, testLabels, classes);
metrics = computeClassificationMetrics(testLabels, TestVotes, classes);
testaccuracy = metrics.accuracy;
fprintf("The Naive-Bayes model's test accuracy is %2.2f percent with a macro F1 score of %.3f in %.3f seconds\n", ...
    testaccuracy, metrics.macroF1, ttTime);

%set up the output
results.modelName = "Naive-Bayes";
results.model = model;
results.predLabels = predLabels;
results.recordVotes = TestVotes;
results.trueLabels = testLabels;
results.testAccuracy = testaccuracy;
results.macroF1 = metrics.macroF1;
results.metrics = metrics;
results.confusionMatrix = metrics.confusionMatrix;
results.classes = classes;
results.time = ttTime;

end
