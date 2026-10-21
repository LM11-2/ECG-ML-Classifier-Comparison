function results = groupedCrossValidation(scatFeatures, windowLabels, recordLabels, Nwin, numFolds, randomSeed)
% Train and evaluate an SVM with record-grouped cross-validation

if nargin < 5 || isempty(numFolds)
    numFolds = 5;
end

if nargin < 6 || isempty(randomSeed)
    error('A random seed is required.');
end

if ~iscategorical(windowLabels)
    windowLabels = categorical(windowLabels);
end

if ~iscategorical(recordLabels)
    recordLabels = categorical(recordLabels);
end

numRecords = numel(recordLabels);
numWindows = size(scatFeatures, 1);

if numWindows ~= numRecords * Nwin
    error('The number of windows must equal the number of records multiplied by Nwin.');
end

rng(randomSeed);
recordCV = cvpartition(recordLabels, KFold=numFolds);
predictedLabels = strings(size(windowLabels));
foldAccuracies = zeros(numFolds, 1);

template = templateSVM(KernelFunction="polynomial", PolynomialOrder=2, ...
    KernelScale="auto", BoxConstraint=1, Standardize=true);

for fold = 1:recordCV.NumTestSets
    trainRecordMask = training(recordCV, fold);
    testRecordMask = test(recordCV, fold);
    trainWindowMask = repelem(trainRecordMask, Nwin);
    testWindowMask = repelem(testRecordMask, Nwin);

    model = fitcecoc(scatFeatures(trainWindowMask, :), windowLabels(trainWindowMask), ...
        Learners=template, Coding="onevsone", ClassNames={'ARR'; 'CHF'; 'NSR'});
    foldPredictions = predict(model, scatFeatures(testWindowMask, :));
    predictedLabels(testWindowMask) = string(foldPredictions);
    foldAccuracies(fold) = mean(foldPredictions == windowLabels(testWindowMask)) * 100;
end

predictedLabels = categorical(predictedLabels, categories(windowLabels));
windowAccuracy = mean(predictedLabels == windowLabels) * 100;

classes = categorical({'ARR', 'CHF', 'NSR'});
[recordVotes, recordCounts] = helperMajorityVote(predictedLabels, recordLabels, classes);
recordAccuracy = mean(recordVotes == recordLabels) * 100;
confMat = confusionmat(recordLabels, recordVotes);

fprintf("The record-grouped cross-validation window accuracy is %2.2f percent.\n", ...
    windowAccuracy);
fprintf("The record-grouped cross-validation record accuracy is %2.2f percent.\n", ...
    recordAccuracy);

results.windowAccuracy = windowAccuracy;
results.recordAccuracy = recordAccuracy;
results.confusionMatrix = confMat;
results.foldAccuracies = foldAccuracies;
results.meanFoldAccuracy = mean(foldAccuracies);
results.stdFoldAccuracy = std(foldAccuracies);
results.predictions = predictedLabels;
results.trueWindowLabels = windowLabels;
results.recordVotes = recordVotes;
results.recordCounts = recordCounts;
results.trueRecordLabels = recordLabels;
results.classes = classes;
results.numFolds = numFolds;
results.Nwin = Nwin;

end
