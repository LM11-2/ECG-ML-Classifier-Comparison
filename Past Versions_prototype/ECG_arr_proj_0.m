%first iteration, followed example and had the ECG data preprocessed

disp("Hello World")
processor = ECGPreprocessor(2);
processor.printDownloadInstruction

%https://www.mathworks.com/help/wavelet/ug/ecg-signal-classification-using-wavelet-time-scattering.html

%%%%%%%%%%%%%%%%%%%%  SETUP AND LOADING DATASET %%%%%%%%%%%%%%%%%%%%%%%%%%

myfold = pwd;
%unzip(fullfile(myfold,"physionet_ECG_data-main.zip"),myfold); %fullfile joins the arguments together to form the file address
%unzip(fullfile(myfold,"physionet_ECG_data-main","ECGData.zip"),fullfile(myfold,"ECGData"));
%^^^^^^commented ts out so it doesnt keep trying to unzip and create new
%folders


load(fullfile(myfold,"ECGData","ECGData.mat"))

%%%%%% EXPLORING ECGData %%%%%%%%%%%
names = fieldnames(ECGData);
size(ECGData.(names{1}));
size(ECGData.(names{2}));
(max(size(names)));

%%----------- TRAIN TEST SPLIT---------------%%
%% SETUP %%
%percent_train = 70;
percent_test = 30;
[X,y] = unpackData(ECGData,"Labels");


% size(X)
% size(y)
% class(X)
% class(y)

[trainData,testData,trainLabels,testLabels] = randomSplit(percent_test,X,y,12);
% size(trainData)

Ctrain = countcats(trainLabels)./numel(trainLabels).*100;
Ctest = countcats(testLabels)./numel(testLabels).*100;


%%----------- PLOT SAMPLES --------------------%%

%helperPlotRandomRecords(ECGData,14);

%%----------- WAVELET TIME SCATTERING --------------------%%

%run the wavelet transform
N = size(X,2);
sn = waveletScattering(SignalLength=N,InvarianceScale=150,SamplingFrequency=128);

%visualize the wavelet filters
[fb,f,filterparams] = filterbank(sn);
figure
tiledlayout(2,1)
nexttile
plot(f,fb{2}.psift)
xlim([0 128])
grid on
title("1st Filter Bank Wavelet Filters")
nexttile
plot(f,fb{3}.psift)
xlim([0 128])
grid on
title("2nd Filter Bank Wavelet Filters")
xlabel("Hz")


%demonstrate invariance scale
figure
phi = ifftshift(ifft(fb{1}.phift));
psiL1 = ifftshift(ifft(fb{2}.psift(:,end)));
t = (-2^15:2^15-1).*1/128;
scalplt = plot(t,phi);
hold on
grid on
ylim([-1.5e-4 1.6e-4])
plot([-75 -75],[-1.5e-4 1.6e-4],"k--")
plot([75 75],[-1.5e-4 1.6e-4],"k--")
xlabel("Seconds")
ylabel("Amplitude")
wavplt = plot(t,[real(psiL1) imag(psiL1)]);
legend([scalplt wavplt(1) wavplt(2)], ...
    {"Scaling Function","Wavelet-Real Part","Wavelet-Imaginary Part"})
title({"Scaling Function";"Coarsest-Scale Wavelet First Filter Bank"})
hold off

%get scattering coefficients for training data
scat_features_train = featureMatrix(sn,trainData');
%make it compatible for the SVM classifier
Nwin = size(scat_features_train,2);
scat_features_train = permute(scat_features_train,[2 3 1]);
scat_features_train  = reshape(scat_features_train,size(scat_features_train,1)*size(scat_features_train,2),[]);

%repeat for the testing data
scat_features_test = featureMatrix(sn,testData');
scat_features_test = permute(scat_features_test,[2 3 1]);
scat_features_test = reshape(scat_features_test,size(scat_features_test,1)*size(scat_features_test,2),[]);

%make labels to match the # of windows
[sequence_labels_train,sequence_labels_test] = createSequenceLabels(Nwin,trainLabels,testLabels);

%Cross Validation with two analyses

%analysis 1: 5-fold cross validation of multiclass SVM classification w/
%quadratic kernel
scat_features = [scat_features_train; scat_features_test];
allLabels_scat = [sequence_labels_train; sequence_labels_test];
rng(1); %so the results are the same each time and we can verify
template = templateSVM(KernelFunction="polynomial",PolynomialOrder=2,KernelScale="auto",BoxConstraint=1,Standardize=true);
classificationSVM = fitcecoc(scat_features,allLabels_scat,Learners=template,Coding="onevsone",ClassNames={'ARR';'CHF';'NSR'});
kfoldmodel = crossval(classificationSVM,KFold=5);

%compute loss and display
predLabels = kfoldPredict(kfoldmodel); %cell of chars
loss = kfoldLoss(kfoldmodel)*100;
predLabels = categorical(predLabels);
confmatCV = confusionmat(allLabels_scat,predLabels)
fprintf("The accuracy is %2.2f percent. \n",100-loss); %the 2.2 means 2 digits before and after decimal

classes = categorical({'ARR','CHF','NSR'});
[ClassVotes,ClassCounts] = helperMajorityVote(predLabels,[trainLabels; testLabels],classes);

CVaccuracy = sum(eq(ClassVotes,categorical([trainLabels;testLabels])))/numel([trainLabels;testLabels])*100;
fprintf("True cross-validation accuracy is %2.2f percent. \n",CVaccuracy);

MVconfmatCV = confusionmat([trainLabels;testLabels],ClassVotes);
MVconfmatCV


%SVM Classification
model = fitcecoc(scat_features_train,sequence_labels_train,Learners=template,Coding="onevsone",ClassNames={'ARR','CHF','NSR'});
predLabels = predict(model,scat_features_test);
[TestVotes,TestCounts] = helperMajorityVote(predLabels,testLabels,classes);
testaccuracy = sum(eq(TestVotes,testLabels))/numel(testLabels)*100;
fprintf("The test accuracy is %2.2f percent. \n",testaccuracy);
confusionchart(testLabels,TestVotes)

%------------------------------------------------%
%------------------------------------------------%
%-----------------Helper Functions --------------%

%Train Test Split function
%Didn't have the helper function so made it myself
%takes the percent test size, features, labels, and random seed
function [trainData,testData,trainLabels,testLabels] = randomSplit(test_size,feats,labels,seed)
%Setup
if nargin < 4 || isempty(seed)
    seed = 12;
end
rng(seed);
if nargin < 1 || isempty(test_size)
    test_size = 0.2;
end

if test_size > 1
    test_size = test_size/100;
end

cv = cvpartition(labels, 'HoldOut', test_size);
trainData = feats(training(cv), :);
testData = feats(test(cv), :);
trainLabels = labels(training(cv));
testLabels = labels(test(cv));
end

%helper function to unpack data into its features, requires knowing name of
%label column first
function [feats,labels] = unpackData(data,labelname)

labels = categorical(data.(labelname));

names = fieldnames(data);
names(strcmp(names,labelname)) = [];
feats = cell(1,numel(names));

for ii = 1:numel(names)
    feats{ii} = data.(names{ii});
end
feats = [feats{:}];
end

%helper function to label the newly scattered windows
function [seq_train_labels,seq_test_labels] = createSequenceLabels(Nwin,trainLabels,testLabels)
seq_train_labels = categorical(repelem(trainLabels(:),Nwin));
seq_test_labels = categorical(repelem(testLabels(:),Nwin));
end


%% HELPER FUNCTIONS FROM EXAMPLE %%
function helperPlotRandomRecords(ECGData,randomSeed)
% This function is only intended to support the XpwWaveletMLExample. It may
% change or be removed in a future release.

if nargin==2
    rng(randomSeed)
end

M = size(ECGData.Data,1);
idxsel = randperm(M,4);
tiledlayout(2,2)
for numplot = 1:4
    nexttile
    plot(ECGData.Data(idxsel(numplot),1:3000))
    ylabel('Volts')
    if numplot > 2
        xlabel('Samples')
    end
    title(ECGData.Labels{idxsel(numplot)})
end

end

function [ClassVotes,ClassCounts] = helperMajorityVote(predLabels,origLabels,classes)
% This function is in support of ECGWaveletTimeScatteringExample. It may
% change or be removed in a future release.

% Make categorical arrays if the labels are not already categorical
predLabels = categorical(predLabels);
origLabels = categorical(origLabels);
% Expects both predLabels and origLabels to be categorical vectors
Npred = numel(predLabels);
Norig = numel(origLabels);
Nwin = Npred/Norig;
predLabels = reshape(predLabels,Nwin,Norig);
assert(size(predLabels,2) == length(origLabels));
ClassCounts = countcats(predLabels);
[~,idx] = max(ClassCounts);
ClassVotes = classes(idx);
% Check for any ties in the maximum values and ensure they are marked as
% error if the mode occurs more than once
modecnt = modecount(predLabels,string(classes));
ClassVotes(modecnt>1) = categorical({'NoUniqueMode'});
ClassVotes = ClassVotes(:);

%-------------------------------------------------------------------------
    function modecnt = modecount(predlabels,classes)
        % Ensure there is a unique mode
        modecnt = zeros(size(predlabels,2),1);
        for nc = 1:size(predlabels,2)
            hc = histcounts(predlabels(:,nc),classes);
            hc = hc-max(hc);
            if sum(hc == 0) > 1
                modecnt(nc) = 1;
            end
        end

    end

end