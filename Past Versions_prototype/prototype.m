clc; close all;
randomSeed = 12;

disp("Loading preprocessed data...");
load('processed4test.mat');


%%%%%% EXPLORING ECGData %%%%%%%%%%%
names = fieldnames(ECGData);
size(ECGData.(names{1}));
size(ECGData.(names{2}));
(max(size(names)));

%%----------- TRAIN TEST SPLIT---------------%%
%% SETUP %%
%percent_train = 70;
percent_test = 30;
[X, y] = unpackData(ECGData, "Labels");


% size(X)
% size(y)
% class(X)
% class(y)

[trainData, testData, trainLabels, testLabels] = randomSplit(percent_test, X, y, randomSeed);
% size(trainData)

Ctrain = countcats(trainLabels) ./ numel(trainLabels) .* 100;
Ctest = countcats(testLabels) ./ numel(testLabels) .* 100;


%%----------- PLOT SAMPLES --------------------%%

%helperPlotRandomRecords(ECGData, randomSeed);

%%----------- WAVELET TIME SCATTERING --------------------%%

%run the wavelet transform
N = size(X, 2);
sn = waveletScattering(SignalLength=N, InvarianceScale=150, SamplingFrequency=128);

%visualize the wavelet filters
[fb, f, filterparams] = filterbank(sn);
figure
tiledlayout(2, 1)
nexttile
plot(f, fb{2}.psift)
xlim([0 128])
grid on
title("1st Filter Bank Wavelet Filters")
nexttile
plot(f, fb{3}.psift)
xlim([0 128])
grid on
title("2nd Filter Bank Wavelet Filters")
xlabel("Hz")


%demonstrate invariance scale
figure
phi = ifftshift(ifft(fb{1}.phift));
psiL1 = ifftshift(ifft(fb{2}.psift(:, end)));
t = (-2^15:2^15-1) .* 1 / 128;
scalplt = plot(t, phi);
hold on
grid on
ylim([-1.5e-4 1.6e-4])
plot([-75 -75], [-1.5e-4 1.6e-4], "k--")
plot([75 75], [-1.5e-4 1.6e-4], "k--")
xlabel("Seconds")
ylabel("Amplitude")
wavplt = plot(t, [real(psiL1) imag(psiL1)]);
legend([scalplt wavplt(1) wavplt(2)], ...
    {"Scaling Function", "Wavelet-Real Part", "Wavelet-Imaginary Part"})
title({"Scaling Function"; "Coarsest-Scale Wavelet First Filter Bank"})
hold off

%get scattering coefficients for training data
scat_features_train = featureMatrix(sn, trainData');
%make it compatible for the SVM classifier
Nwin = size(scat_features_train, 2);
scat_features_train = permute(scat_features_train, [2 3 1]);
scat_features_train = reshape(scat_features_train, size(scat_features_train, 1) * size(scat_features_train, 2), []);

%repeat for the testing data
scat_features_test = featureMatrix(sn, testData');
scat_features_test = permute(scat_features_test, [2 3 1]);
scat_features_test = reshape(scat_features_test, size(scat_features_test, 1) * size(scat_features_test, 2), []);

%make labels to match the # of windows
[sequence_labels_train, sequence_labels_test] = createSequenceLabels(Nwin, trainLabels, testLabels);

%Cross Validation with two analyses
%analysis 1: 5-fold cross validation of multiclass SVM classification w/
%quadratic kernel
scat_features = [scat_features_train; scat_features_test];
allLabels_scat = [sequence_labels_train; sequence_labels_test];
rng(randomSeed); %so the results are the same each time and we can verify
template = templateSVM(KernelFunction="polynomial", PolynomialOrder=2, KernelScale="auto", BoxConstraint=1, Standardize=true);
classificationSVM = fitcecoc(scat_features, allLabels_scat, Learners=template, Coding="onevsone", ClassNames={'ARR'; 'CHF'; 'NSR'});
kfoldmodel = crossval(classificationSVM, KFold=5);

%compute loss and display
predLabels = kfoldPredict(kfoldmodel); %cell of chars
loss = kfoldLoss(kfoldmodel) * 100;
predLabels = categorical(predLabels);
confmatCV = confusionmat(allLabels_scat, predLabels)
fprintf("The accuracy is %2.2f percent. \n", 100-loss); %the 2.2 means 2 digits before and after decimal

classes = categorical({'ARR', 'CHF', 'NSR'});
[ClassVotes, ClassCounts] = helperMajorityVote(predLabels, [trainLabels; testLabels], classes);

CVaccuracy = sum(eq(ClassVotes, categorical([trainLabels; testLabels]))) / numel([trainLabels; testLabels]) * 100;
fprintf("True cross-validation accuracy is %2.2f percent. \n", CVaccuracy);

MVconfmatCV = confusionmat([trainLabels; testLabels], ClassVotes);
MVconfmatCV

%{
---------------------------------------------------
NOTE: This classification is likely to be inflated
Since all the windows in scat_features are treated as separate samples, it
is possible for windows of the same ECG record to be in training and
testing for a single fold.
The model will see data that is very similar, and this could inflate the
cross-validation accuracy.
---------------------------------------------------
%}




SVMresults = SVMClassifier(scat_features_train, scat_features_test, sequence_labels_train, testLabels, {'ARR', 'CHF', 'NSR'});
RandForestresults = RandForestClassifier(scat_features_train, scat_features_test, sequence_labels_train, testLabels, {'ARR', 'CHF', 'NSR'});
KNNresults = KNNClassifier(scat_features_train, scat_features_test, sequence_labels_train, testLabels, {'ARR', 'CHF', 'NSR'});
NaiveBayesresults = NaiveBayesClassifier(scat_features_train, scat_features_test, sequence_labels_train, testLabels, {'ARR', 'CHF', 'NSR'});
FNNetresults = NeuralNetClassifier4(scat_features_train, scat_features_test, sequence_labels_train, testLabels, {'ARR', 'CHF', 'NSR'}, randomSeed);

%resultsTable = displayResults({SVMresults, RandForestresults, KNNresults, NaiveBayesresults, FNNetresults});
%plotConfusionMatrices({SVMresults, RandForestresults, KNNresults, NaiveBayesresults, FNNetresults})
plotNNTrainingCurves(FNNetresults)



%% HELPER FUNCTION FROM MATLAB EXAMPLE %%
function helperPlotRandomRecords(ECGData, randomSeed)
% This function is only intended to support the XpwWaveletMLExample. It may
% change or be removed in a future release.

if nargin == 2
    rng(randomSeed)
end

M = size(ECGData.Data, 1);
idxsel = randperm(M, 4);
tiledlayout(2, 2)
for numplot = 1:4
    nexttile
    plot(ECGData.Data(idxsel(numplot), 1:3000))
    ylabel('Volts')
    if numplot > 2
        xlabel('Samples')
    end
    title(ECGData.Labels{idxsel(numplot)})
end

end
