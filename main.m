randomSeed = 95;
processor = ECGPreprocessor(randomSeed); % Seeded for reproducibility

%%%%%%%%%%%%%%%%%%%%  SETUP AND LOADING DATASET %%%%%%%%%%%%%%%%%%%%%%%%%%
disp("Please select the folder containing the Raw ECG Signal Data");
parentFolder = uigetdir(pwd, 'Select the Raw-Signals folder');

if isequal(parentFolder, 0) || isempty(parentFolder)
    error('No raw-signal folder was selected.');
end

dbConfig = processor.createConfig({'mitdb', 'chfdb', 'nsrdb'}, [360, 250, 128], {'ARR', 'CHF', 'NSR'});
ECGData = processor.preprocessFromFolder(parentFolder, dbConfig, 128, 65536);

outputFolder = fullfile(pwd, 'Outputs');
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

outputFile = fullfile(outputFolder, 'processed4test.mat');
save(outputFile, "ECGData");
fprintf('Data saved to %s\n', outputFile);


%%%%%% EXPLORING ECGData %%%%%%%%%%%
% names = fieldnames(ECGData);
% size(ECGData.(names{1}));
% size(ECGData.(names{2}));
% (max(size(names)));

%%----------- TRAIN TEST SPLIT---------------%%
%% SETUP %%
%percent_train = 70;
percent_test = 30;
[X, y] = unpackData(ECGData, "Labels");

[trainData, testData, trainLabels, testLabels] = randomSplit(percent_test, X, y, randomSeed);

Ctrain = countcats(trainLabels) ./ numel(trainLabels) .* 100;
Ctest = countcats(testLabels) ./ numel(testLabels) .* 100;


%%----------- PLOT SAMPLES --------------------%%

helperPlotRandomRecords(ECGData, randomSeed);

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

%make labels to match the number of windows
[sequence_labels_train, sequence_labels_test] = createSequenceLabels(Nwin, trainLabels, testLabels);

scat_features = [scat_features_train; scat_features_test];
allLabels_scat = [sequence_labels_train; sequence_labels_test];
allRecordLabels = [trainLabels; testLabels];
CVresults = groupedCrossValidation(scat_features, allLabels_scat, ...
    allRecordLabels, Nwin, 5, randomSeed);

SVMresults = SVMClassifier(scat_features_train, scat_features_test, sequence_labels_train, testLabels, {'ARR', 'CHF', 'NSR'});
RandForestresults = RandForestClassifier(scat_features_train, scat_features_test, sequence_labels_train, testLabels, {'ARR', 'CHF', 'NSR'},randomSeed);
KNNresults = KNNClassifier(scat_features_train, scat_features_test, sequence_labels_train, testLabels, {'ARR', 'CHF', 'NSR'});
NaiveBayesresults = NaiveBayesClassifier(scat_features_train, scat_features_test, sequence_labels_train, testLabels, {'ARR', 'CHF', 'NSR'});
FNNetresults = FeedForwardNeuralNetClassifier(scat_features_train, scat_features_test, ...
    sequence_labels_train, testLabels, {'ARR', 'CHF', 'NSR'}, Nwin, randomSeed);

resultsTable = displayResults({SVMresults, RandForestresults, KNNresults, NaiveBayesresults, FNNetresults});
plotConfusionMatrices({SVMresults, RandForestresults, KNNresults, NaiveBayesresults, FNNetresults})
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
