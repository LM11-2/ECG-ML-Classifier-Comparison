function resultsTable = displayResults(allResults)

%takes cell array of results struct for each classifier
%returns table with results ranked by accuracy

numModels = numel(allResults);

modelNames = cell(numModels, 1);
testAccuracies = zeros(numModels, 1);
trainingTimes = zeros(numModels, 1);

for i = 1:numModels
    modelNames{i} = allResults{i}.modelName;
    testAccuracies(i) = allResults{i}.testAccuracy;
    trainingTimes(i) = allResults{i}.time;
end

resultsTable = table(modelNames, testAccuracies, trainingTimes, ...
    'VariableNames', {'Model', 'Accuracy_Percent', 'Time_Seconds'});

resultsTable = sortrows(resultsTable, 'Accuracy_Percent', 'descend');

fprintf("\n");
fprintf("╔══════════════════════════════════════════════════════════════════════════════════╗\n");
fprintf("║                      ECG CLASSIFICATION - RESULTS SUMMARY                        ║\n");
fprintf("╠══════════════════════════════════════════════════════════════════════════════════╣\n");
fprintf("║  Rank  │  Model                                       │  Accuracy  │  Time       ║\n");
fprintf("╠══════════════════════════════════════════════════════════════════════════════════╣\n");

for i = 1:numModels
    fprintf("║   %d    │  %-44s │   %6.2f%%  │  %6.2fs   ║\n", ...
        i, resultsTable.Model{i}, resultsTable.Accuracy_Percent(i), resultsTable.Time_Seconds(i));
end

fprintf("╚══════════════════════════════════════════════════════════════════════════════════╝\n");
fprintf("\n");

fprintf("Most Accurate Model: %s (%.2f%% accuracy)\n\n", ...
    resultsTable.Model{1}, resultsTable.Accuracy_Percent(1));

[fastestTime, fastestIndex] = min(resultsTable.Time_Seconds);
fprintf("Fastest Model: %s (%.2f seconds)\n\n", ...
    resultsTable.Model{fastestIndex}, fastestTime);

end
