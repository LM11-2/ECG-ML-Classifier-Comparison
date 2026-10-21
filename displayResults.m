function resultsTable = displayResults(allResults)

numModels = numel(allResults);
modelNames = cell(numModels, 1);
testAccuracies = zeros(numModels, 1);
macroF1Scores = zeros(numModels, 1);
trainingTimes = zeros(numModels, 1);

for ii = 1:numModels
    modelNames{ii} = allResults{ii}.modelName;
    testAccuracies(ii) = allResults{ii}.testAccuracy;
    macroF1Scores(ii) = allResults{ii}.macroF1;
    trainingTimes(ii) = allResults{ii}.time;
end

resultsTable = table(modelNames, testAccuracies, macroF1Scores, trainingTimes, ...
    VariableNames={'Model', 'Accuracy_Percent', 'Macro_F1', 'Time_Seconds'});
resultsTable = sortrows(resultsTable, 'Macro_F1', 'descend');

fprintf("\nECG CLASSIFICATION - RESULTS SUMMARY\n\n");
disp(resultsTable)

fprintf("Highest Macro F1 Score: %s (%.3f)\n\n", ...
    resultsTable.Model{1}, resultsTable.Macro_F1(1));

[fastestTime, fastestIndex] = min(resultsTable.Time_Seconds);
fprintf("Fastest Model: %s (%.2f seconds)\n\n", ...
    resultsTable.Model{fastestIndex}, fastestTime);

end
