classdef ECGPreprocessor
    properties
        seed;
    end

    properties (Constant)
        %defaults
        DEFAULTFREQ = 128;
        DEFAULTLEN = 2^16;
        DEFAULTSEED = 12;
    end

    methods
        %constructor
        function obj = ECGPreprocessor(seed)
            if nargin < 1 || isempty(seed)
                seed = obj.DEFAULTSEED;
            end
            obj.seed = seed;
        end

        %{
        Processing function
        Output: preprocessed: full ECG data file with data and labels
        Input: rawFolder: folder in which the databases are stored
               dbConfig: struct with info for each database
               targetFreq: target sampling frequency
               targetLen: target # of samples
        %}
        function preprocessed = preprocessFromFolder(obj, rawFolder, dbConfig, targetFreq, targetLen)
            if nargin < 4 || isempty(targetFreq)
                targetFreq = obj.DEFAULTFREQ;
            end
            if nargin < 5 || isempty(targetLen)
                targetLen = obj.DEFAULTLEN;
            end

            allData = [];
            allLabels = {};

            databases = fieldnames(dbConfig);

            for ii = 1:length(databases)
                dbName = databases{ii};
                dbFolder = fullfile(rawFolder, dbName);

                if ~isfolder(dbFolder)
                    error('Required database folder not found: %s', dbFolder);
                end

                originalFreq = dbConfig.(dbName).freq;
                label = dbConfig.(dbName).label;

                fprintf('Processing %s (label: %s, %d Hz → %d Hz, %d samples)...\n', ...
                    dbName, label, originalFreq, targetFreq, targetLen);

                [data, labels] = obj.processDatabase(dbFolder, originalFreq, label, targetFreq, targetLen);
                allData = [allData; data];
                allLabels = [allLabels; labels];
            end

            requiredLabels = {'ARR', 'CHF', 'NSR'};
            for ii = 1:numel(requiredLabels)
                if ~any(strcmp(allLabels, requiredLabels{ii}))
                    error('No usable %s samples were found after preprocessing.', requiredLabels{ii});
                end
            end

            preprocessed.Data = allData;
            preprocessed.Labels = allLabels;

            fprintf('Final size: %d records x %d samples\n', ...
                size(allData, 1), size(allData, 2));
        end
    end

    methods (Access = private)
        %{
        process individual database
        Output: data: [N x targetLen] matrix of processed signals
                labels: [N x 1] matrix of labels
        Input: dbFolder: folder with .dat file
               label: class label assigned to all records
               targetFreq: target sampling frequency
               targetLen: target # of samples
        %}
        function [data, labels] = processDatabase(obj, dbFolder, originalFreq, label, targetFreq, targetLen)

            if nargin < 5 || isempty(targetFreq)
                targetFreq = obj.DEFAULTFREQ;
            end
            if nargin < 6 || isempty(targetLen)
                targetLen = obj.DEFAULTLEN;
            end

            records = dir(fullfile(dbFolder, '*.dat'));

            if isempty(records)
                error('No usable records were found in %s because it contains no .dat files.', dbFolder);
            end

            data = [];
            labels = {};

            for ii = 1:length(records)
                recordName = erase(records(ii).name, '.dat');
                recordPath = fullfile(dbFolder, recordName);

                try
                    processedData = obj.processRecord(recordPath, originalFreq, targetFreq, targetLen);

                    %channels split into different records
                    numChannels = size(processedData, 1);
                    for ch = 1:numChannels
                        data = [data; processedData(ch, :)];
                        labels = [labels; {label}];
                    end
                catch ME
                    warning('Error processing %s: %s', recordName, ME.message);
                end
            end

            if isempty(data)
                error('No usable records were produced from %s.', dbFolder);
            end

            fprintf('Processed %d records from %s\n', size(data, 1), dbFolder);
        end

        %{
        process a single record
        Output: processedData: [channels x targetLen] matrix
         Input: recordPath: path to record (w/out file extension)
                originalFreq: OG sampling frequency
                targetFreq: target sampling frequency
                targetLen: target # of samples
        %}
        function processedData = processRecord(obj, recordPath, originalFreq, targetFreq, targetLen)

            if nargin < 4 || isempty(targetFreq)
                targetFreq = obj.DEFAULTFREQ;
            end
            if nargin < 5 || isempty(targetLen)
                targetLen = obj.DEFAULTLEN;
            end

            %this is where the preprocessing steps are applied
            %1. scaling
            %2. resample to target frequency
            %3. Separate the channels
            %4. Truncate to desired length

            %trying to change directory so rdsamp can find the folders
            originalDir = pwd;
            [folder, recordName, ~] = fileparts(recordPath);

            try
                if ~isempty(folder)
                    cd(folder);
                end
                [signal, ~, ~] = rdsamp(recordName);
                cd(originalDir);
            catch ME
                cd(originalDir); %need to go back regardless
                rethrow(ME);
            end

            numChannels = size(signal, 2);

            if originalFreq ~= targetFreq
                [P, Q] = rat(targetFreq / originalFreq); %rational fraction approx.
                newLen = ceil(size(signal, 1) * P / Q); %round towards +inf
                resampledSig = zeros(newLen, numChannels);

                for ch = 1:numChannels
                    resampledSig(:, ch) = resample(signal(:, ch), P, Q);
                end
            else
                resampledSig = signal;
            end

            processedData = resampledSig';

            currentLen = size(processedData, 2);
            if currentLen >= targetLen
                processedData = processedData(:, 1:targetLen);
            else
                warning('Record is shorter than target, data will be padded with zeros')
                padding = zeros(numChannels, targetLen - currentLen);
                processedData = [processedData, padding];
            end
        end
    end

    methods (Static)

        %{
        Helper to create database config struct
        Output: config: struct to be used with preprocessFromFolder
         Input: dbNames: Cell array of db folder names
                freqs: array of OG sampling frequencies
                labels: cell array of class labels
        %}
        function config = createConfig(dbNames, freqs, labels)
            if length(dbNames) ~= length(freqs) || length(dbNames) ~= length(labels)
                error('All inputs must have same length');
            end

            config = struct();
            for ii = 1:length(dbNames)
                config.(dbNames{ii}) = struct('freq', freqs(ii), 'label', labels{ii});
            end
        end

    end

end
