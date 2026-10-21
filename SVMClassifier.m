%{
Xtrain: training features
Xtest: testing features
ytrain: training labels 
ytest: testing labels
Nwin: # of windows per record
^window level
testLabels: test labels (record level)
classes: array of class names

%}


function results = SVMClassifier(Xtrain, Xtest, ytrain, testLabels, classes)

if iscell(classes)
    classes = categorical(classes);
end

template = templateSVM(KernelFunction="polynomial", PolynomialOrder=2, KernelScale="auto", BoxConstraint=1, Standardize=true);
classNamesCell = cellstr(classes);

tic; %start timer when training starts
model = fitcecoc(Xtrain, ytrain, Learners=template, Coding="onevsone", ClassNames=classNamesCell);
predLabels = predict(model, Xtest);
ttTime = toc;

[TestVotes, TestCounts] = helperMajorityVote(predLabels, testLabels, classes);
metrics = computeClassificationMetrics(testLabels, TestVotes, classes);
testaccuracy = metrics.accuracy;
fprintf("The SVM model's test accuracy is %2.2f percent with a macro F1 score of %.3f in %.3f seconds\n", ...
    testaccuracy, metrics.macroF1, ttTime);

%set up the output
results.modelName = "SVM";
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
