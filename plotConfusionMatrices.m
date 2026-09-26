function plotConfusionMatrices(allResults)
% All confusion matrices plotted together

numModels = numel(allResults);

numCols = ceil(sqrt(numModels));
numRows = ceil(numModels / numCols);

figure('Name', 'Confusion Matrices', ...
    'Position', [50, 50, 400 * numCols, 350 * numRows], ...
    'Color', 'w');

for i = 1:numModels
    subplot(numRows, numCols, i);

    cm = confusionchart(allResults{i}.trueLabels, allResults{i}.recordVotes);

    cm.Title = allResults{i}.modelName;
    cm.FontSize = 9;
    cm.RowSummary = 'row-normalized';
    cm.ColumnSummary = 'column-normalized';
end

sgtitle('ECG Classification: Confusion Matrices', 'FontSize', 14, 'FontWeight', 'bold');

end
