%helper function to unpack data into its features, requires knowing name of
%label column first
function [feats, labels] = unpackData(data, labelname)

labels = categorical(data.(labelname));

names = fieldnames(data);
names(strcmp(names, labelname)) = [];
feats = cell(1, numel(names));

for ii = 1:numel(names)
    feats{ii} = data.(names{ii});
end
feats = [feats{:}];
end
