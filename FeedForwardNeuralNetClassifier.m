function results = FeedForwardNeuralNetClassifier(Xtrain, Xtest, ytrain, testLabels, classes, Nwin, randomSeed)
% NEURALNETCLASSIFIER Train and evaluate a feedforward neural network

if iscell(classes)
    classes = categorical(classes);
end

%trainNetwork requires input to be categorical variable
if ~iscategorical(ytrain)
    ytrain = categorical(ytrain);
end

if nargin < 7 || isempty(randomSeed)
    error('A random seed is required.');
end

if size(Xtrain, 1) ~= numel(ytrain) || mod(numel(ytrain), Nwin) ~= 0
    error('Training features and labels must contain the same number of complete records.');
end

recordLabels = ytrain(1:Nwin:end);
rng(randomSeed);
validationSplit = cvpartition(recordLabels, HoldOut=0.2);
trainWindowMask = repelem(training(validationSplit), Nwin);
validationWindowMask = repelem(test(validationSplit), Nwin);

Xval = Xtrain(validationWindowMask, :);
Yval = ytrain(validationWindowMask);
Xtrain = Xtrain(trainWindowMask, :);
ytrain = ytrain(trainWindowMask);

%dimensions
numFeatures = size(Xtrain, 2);
numClasses = numel(categories(classes));

% Define network architecture
layers = [
    featureInputLayer(numFeatures)
    fullyConnectedLayer(16)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(8)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
    ];

% Training options
options = trainingOptions('adam', ...
    MaxEpochs=50, ...
    MiniBatchSize=32, ...
    InitialLearnRate=0.001, ...
    ValidationData={Xval, Yval}, ...
    ValidationFrequency=10, ...
    ValidationPatience=10, ...
    Verbose=false, ...
    Plots='none');

%%%%%%%%%%%%
tic;
[model, trainInfo] = trainNetwork(Xtrain, ytrain, layers, options);
predLabels = classify(model, Xtest);
ttTime = toc;

[TestVotes, TestCounts] = helperMajorityVote(predLabels, testLabels, classes);
metrics = computeClassificationMetrics(testLabels, TestVotes, classes);
testaccuracy = metrics.accuracy;
fprintf("The FeedForward Neural Network's test accuracy is %2.2f percent with a macro F1 score of %.3f in %.3f seconds\n", ...
    testaccuracy, metrics.macroF1, ttTime);

results.modelName = 'Feed Forward Neural Network (16-8-3)';
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
results.trainInfo = trainInfo;
results.validationRecordLabels = recordLabels(test(validationSplit));

end
