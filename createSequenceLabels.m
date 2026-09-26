%helper function to label the newly scattered windows
function [seq_train_labels, seq_test_labels] = createSequenceLabels(Nwin, trainLabels, testLabels)
seq_train_labels = categorical(repelem(trainLabels(:), Nwin));
seq_test_labels = categorical(repelem(testLabels(:), Nwin));
end
