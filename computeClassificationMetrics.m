function metrics = computeClassificationMetrics(trueLabels, predictedLabels, classes)

if iscell(classes)
    classes = categorical(classes);
end

%consider NoUniqueMode to classes-possible output from helperMajorityVote
extendedClasses = classes;
if ~ismember("NoUniqueMode",categories(classes))
    extendedClasses = [classes,categorical("NoUniqueMode")];
end

if ~iscategorical(trueLabels)
    trueLabels = categorical(trueLabels);
end

if ~iscategorical(predictedLabels)
    predictedLabels = categorical(predictedLabels);
end

if numel(trueLabels) ~= numel(predictedLabels)
    error('True and predicted labels must have the same number of elements.');
end

numClasses = numel(extendedClasses);
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1 = zeros(numClasses, 1);

for ii = 1:numClasses
    truePos = sum(trueLabels == extendedClasses(ii) & predictedLabels == extendedClasses(ii));
    falsePos = sum(trueLabels ~= extendedClasses(ii) & predictedLabels == extendedClasses(ii));
    falseNeg = sum(trueLabels == extendedClasses(ii) & predictedLabels ~= extendedClasses(ii));

    if truePos + falsePos > 0
        precision(ii) = truePos / (truePos + falsePos);
    end

    if truePos + falseNeg > 0
        recall(ii) = truePos / (truePos + falseNeg);
    end

    if precision(ii) + recall(ii) > 0
        f1(ii) = 2 * precision(ii) * recall(ii) / (precision(ii) + recall(ii));
    end
end

metrics.confusionMatrix = confusionmat(trueLabels, predictedLabels, Order=extendedClasses);
metrics.precision = precision;
metrics.recall = recall;
metrics.f1 = f1;

%remove NoUniqueMode from macro scores
realClassIdx = ~strcmp(string(extendedClasses),"NoUniqueMode");
metrics.macroPrecision = mean(precision(realClassIdx));
metrics.macroRecall = mean(recall(realClassIdx));
metrics.macroF1 = mean(f1(realClassIdx));
metrics.accuracy = mean(trueLabels == predictedLabels) * 100;

end
