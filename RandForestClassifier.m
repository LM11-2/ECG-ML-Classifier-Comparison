function results = RandForestClassifier(Xtrain, Xtest, ytrain, testLabels, classes, randomSeed, numTrees)

if iscell(classes)
    classes = categorical(classes);
end

if (nargin < 6) || isempty(randomSeed)
    randomSeed = 12;
end
if (nargin < 7) || isempty(numTrees)
    numTrees = 100;
end

rng(randomSeed);
treeTemplate = templateTree(Reproducible = true);

tic; %start timer when training starts
model = fitcensemble(Xtrain, ytrain, Method = 'Bag', NumLearningCycles=numTrees, Learners=treeTemplate);
predLabels = predict(model, Xtest);
ttTime = toc;

[TestVotes, TestCounts] = helperMajorityVote(predLabels, testLabels, classes);
metrics = computeClassificationMetrics(testLabels, TestVotes, classes);
testaccuracy = metrics.accuracy;
fprintf("The Random Forest model's test accuracy is %2.2f percent with a macro F1 score of %.3f in %.3f seconds\n", ...
    testaccuracy, metrics.macroF1, ttTime);

%set up the output
results.modelName = "Random Forest";
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
