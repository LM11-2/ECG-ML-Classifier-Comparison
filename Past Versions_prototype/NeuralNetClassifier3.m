function results = NeuralNetClassifier3(Xtrain, Xtest, ytrain, testLabels, classes)
% NEURALNETCLASSIFIER Train and evaluate a feedforward neural network

if iscell(classes)
    classes = categorical(classes);
end

%trainNetwork requires input to be categorical variable
if ~iscategorical(ytrain)
    ytrain = categorical(ytrain);
end

%split into validation and train data to check about overfitting
rng(1);
numTrain = size(Xtrain, 1);
numVal = round(0.2 * numTrain);
idx = randperm(numTrain);

Xval = Xtrain(idx(1:numVal), :);
Yval = ytrain(idx(1:numVal));
Xtrain = Xtrain(idx(numVal + 1:end), :);
ytrain = ytrain(idx(numVal + 1:end));

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
    Plots='training-progress');

%%%%%%%%%%%%
tic;
[model, trainInfo] = trainNetwork(Xtrain, ytrain, layers, options);
predLabels = classify(model, Xtest);
ttTime = toc;

[TestVotes, TestCounts] = helperMajorityVote(predLabels, testLabels, classes);
testaccuracy = sum(eq(TestVotes, testLabels)) / numel(testLabels) * 100;
fprintf("\n===== Feed Forward Neural Network Results =====\n");
fprintf("Test accuracy:       %.2f%%\n", testaccuracy);
fprintf("Training accuracy:   %.2f%%\n", trainInfo.TrainingAccuracy(end));
fprintf("Validation accuracy: %.2f%%\n", trainInfo.ValidationAccuracy(end));
fprintf("Training loss:       %.4f\n", trainInfo.TrainingLoss(end));
fprintf("Validation loss:     %.4f\n", trainInfo.ValidationLoss(end));
fprintf("Time:                %.2f seconds\n", ttTime);
fprintf("==================================\n");

confMat = confusionmat(testLabels, TestVotes);

results.modelName = 'Feed Forward Neural Network (16-8-3)';
results.model = model;
results.predLabels = predLabels;
results.recordVotes = TestVotes;
results.trueLabels = testLabels;
results.testAccuracy = testaccuracy;
results.confusionMatrix = confMat;
results.classes = classes;
results.time = ttTime;
results.trainInfo = trainInfo;

end
