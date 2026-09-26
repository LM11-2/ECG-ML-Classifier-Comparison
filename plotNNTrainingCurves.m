function plotNNTrainingCurves(nnResults)
%takes the FF Neural Network results
%creates a plot of the training curves for the FF Neural Network

if ~isfield(nnResults, 'trainInfo')
    error('Results struct does not contain trainInfo. Make sure you are passing NeuralNet results.');
end

trainInfo = nnResults.trainInfo;

figure('Name', 'Neural Network Training Progress', ...
    'Position', [100, 100, 1800, 500], ...
    'Color', 'w');

%% Plot 1: Accuracy
subplot(1, 2, 1);

% Training accuracy
plot(trainInfo.TrainingAccuracy, 'b-', 'LineWidth', 0.8);
hold on;

% Validation accuracy
if isfield(trainInfo, 'ValidationAccuracy') && ~isempty(trainInfo.ValidationAccuracy)
    valAcc = trainInfo.ValidationAccuracy;
    valIdx = find(~isnan(valAcc));
    plot(valIdx, valAcc(valIdx), 'r-', 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 4);
end
hold off;

xlabel('Iteration', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Accuracy (%)', 'FontSize', 11, 'FontWeight', 'bold');
title('Training vs Validation Accuracy', 'FontSize', 12, 'FontWeight', 'bold');
legend('Training', 'Validation', 'Location', 'southeast', 'FontSize', 10);
grid on;
set(gca, 'GridAlpha', 0.3, 'LineWidth', 1.1);
ylim([0, 100]);

%% Plot 2: Loss
subplot(1, 2, 2);

% Training loss
plot(trainInfo.TrainingLoss, 'b-', 'LineWidth', 0.8);
hold on;

% Validation los
if isfield(trainInfo, 'ValidationLoss') && ~isempty(trainInfo.ValidationLoss)
    valLoss = trainInfo.ValidationLoss;
    valIdx = find(~isnan(valLoss));
    plot(valIdx, valLoss(valIdx), 'r-', 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 4);
end
hold off;

xlabel('Iteration', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Loss', 'FontSize', 11, 'FontWeight', 'bold');
title('Training vs Validation Loss', 'FontSize', 12, 'FontWeight', 'bold');
legend('Training', 'Validation', 'Location', 'northeast', 'FontSize', 10);
grid on;
set(gca, 'GridAlpha', 0.3, 'LineWidth', 1.1);

sgtitle([nnResults.modelName ' - Training Progress'], 'FontSize', 14, 'FontWeight', 'bold');

end
