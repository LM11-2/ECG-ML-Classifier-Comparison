function results = NeuralNetClassifier2(Xtrain, Xtest, ytrain, testLabels, classes)
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
Xtrain = Xtrain(idx(numVal+1:end), :);
ytrain = ytrain(idx(numVal+1:end));


%dimensions
numFeatures = size(Xtrain, 2);
numClasses = numel(categories(classes));

% Define network architecture
layers = [
    featureInputLayer(numFeatures)
    fullyConnectedLayer(128)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(64)
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
    ValidationData={Xval,Yval}, ...
    ValidationFrequency=10, ...
    ValidationPatience=10, ...
    Verbose=false, ...
    Plots='training-progress');




%%%%%%%%%%%%
tic;
model = trainNetwork(Xtrain, ytrain, layers, options);
predLabels = classify(model, Xtest);
ttTime = toc;


[TestVotes, TestCounts] = helperMajorityVote(predLabels, testLabels, classes);
testaccuracy = sum(eq(TestVotes, testLabels)) / numel(testLabels) * 100;
fprintf("The test accuracy is %2.2f percent.\n", testaccuracy);
fprintf("The Feed Forward Neural Netowrk model's test accuracy is %2.2f percent. \n in %.3f seconds",testaccuracy,ttTime);


confMat = confusionmat(testLabels, TestVotes);


results.modelName = 'Neural Network';
results.model = model;
results.predLabels = predLabels;
results.recordVotes = TestVotes;
results.testAccuracy = testaccuracy;
results.confusionMatrix = confMat;
results.classes = classes;
results.time = ttTime;

end