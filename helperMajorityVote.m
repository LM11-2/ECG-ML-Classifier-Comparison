%% HELPER FUNCTION FROM MATLAB EXAMPLE %%

function [ClassVotes, ClassCounts] = helperMajorityVote(predLabels, origLabels, classes)
% This function is in support of ECGWaveletTimeScatteringExample. It may
% change or be removed in a future release.

% Make categorical arrays if the labels are not already categorical
predLabels = categorical(predLabels);
origLabels = categorical(origLabels);
% Expects both predLabels and origLabels to be categorical vectors
Npred = numel(predLabels);
Norig = numel(origLabels);
Nwin = Npred / Norig;
predLabels = reshape(predLabels, Nwin, Norig);
assert(size(predLabels, 2) == length(origLabels));
ClassCounts = countcats(predLabels);
[~, idx] = max(ClassCounts);
ClassVotes = classes(idx);
% Check for any ties in the maximum values and ensure they are marked as
% error if the mode occurs more than once
modecnt = modecount(predLabels, string(classes));
ClassVotes(modecnt > 1) = categorical({'NoUniqueMode'});
ClassVotes = ClassVotes(:);

%-------------------------------------------------------------------------
    function modecnt = modecount(predlabels, classes)
        % Ensure there is a unique mode
        modecnt = zeros(size(predlabels, 2), 1);
        for nc = 1:size(predlabels, 2)
            hc = histcounts(predlabels(:, nc), classes);
            hc = hc - max(hc);
            if sum(hc == 0) > 1
                modecnt(nc) = 1;
            end
        end
    end

end
