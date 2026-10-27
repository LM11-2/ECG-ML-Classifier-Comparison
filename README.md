# ECG Signal Classification with Wavelet Time Scattering

> My personal project comparing machine-learning classifiers for electrocardiogram (ECG) signal classification.

## Overview

This project compares several machine-learning classifiers for distinguishing ECG signals across three classes:

- **ARR** — arrhythmia
- **CHF** — congestive heart failure
- **NSR** — normal sinus rhythm

The workflow preprocesses raw ECG recordings, extracts features with a wavelet time scattering transform, trains multiple classifiers, and compares their performance. The models included are a support vector machine, random forest, k-nearest neighbors, Naive Bayes classifier, and feedforward neural network.

This is an independent student portfolio project that grew out of the ideas in MathWorks' [Wavelet Time Scattering for ECG Signal Classification](https://www.mathworks.com/help/wavelet/ug/ecg-signal-classification-using-wavelet-time-scattering.html) example.

## Project Goals

- Build an end-to-end ECG classification workflow in MATLAB.
- Compare traditional machine-learning models with a feedforward neural network.
- Explore wavelet time scattering as a robust ECG feature-extraction method.
- Practice reproducible experimentation through centralized random seeding.
- Familiarize myself with MATLAB

## Workflow

1. Select a folder containing the raw ECG recordings.
2. Preprocess recordings to a common sampling rate and length.
3. Split the recordings into training and test sets.
4. Extract wavelet time scattering features.
5. Train and evaluate each classifier.
6. Aggregate window-level predictions into record-level predictions using majority vote.
7. Compare results with summary tables, confusion matrices, and neural-network training curves.

## Repository Guide

| File | Purpose |
| --- | --- |
| `main.m` | Main entry point. Selects the raw-signal folder, runs preprocessing, extracts features, trains models, and displays results. |
| `ECGPreprocessor.m` | Class for loading and standardizing ECG recordings from the selected data folder. |
| `SVMClassifier.m` | Trains and evaluates the support vector machine classifier. |
| `RandForestClassifier.m` | Trains and evaluates the random forest classifier. |
| `KNNClassifier.m` | Trains and evaluates the k-nearest neighbors classifier. |
| `NaiveBayesClassifier.m` | Trains and evaluates the Naive Bayes classifier. |
| `NeuralNetClassifier4.m` | Trains and evaluates the feedforward neural network. |
| `randomSplit.m` | Creates a reproducible training/test split. |
| `createSequenceLabels.m` | Matches labels to wavelet-scattering time windows. |
| `helperMajorityVote.m` | Combines window-level predictions into one prediction per ECG record. |
| `displayResults.m` | Ranks and prints classifier results. |
| `plotConfusionMatrices.m` | Creates confusion-matrix visualizations for each model. |
| `plotNNTrainingCurves.m` | Visualizes neural-network training and validation progress. |
| `DownloadInstructions.txt` | Explains how to download and organize the required raw ECG data. |
| `Analysis.txt` | Notes on analysis, validation, and limitations. |
| `Past Versions/` | Archived project iterations and unused classifier versions. |

## Requirements

This project was developed in MATLAB. Before running it, install or configure:

- MATLAB **[add tested release, for example: R20XXx]**
- Wavelet Toolbox
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox
- Signal Processing Toolbox
- A MATLAB-compatible WFDB reader that provides `rdsamp` **[add installation link or instructions]**

## Data Setup

Raw ECG data is not required to be stored in the repository. Follow [DownloadInstructions.txt](DownloadInstructions.txt) to download and arrange the source data.

When prompted by `main.m`, select the parent folder containing these subfolders:

```text
your-data-folder/
├── mitdb/
├── chfdb/
└── nsrdb/
```

The source data originates from PhysioNet databases. Preserve the required data licenses and attributions when obtaining or using the files.

## How to Run

1. Clone or download this repository.
2. Install the required MATLAB toolboxes and configure `rdsamp`.
3. Open MATLAB in the project folder.
4. Run `main.m`.
5. Select the parent folder containing the raw ECG data when prompted.
6. Review the printed model summary and generated visualizations.

The random seed is declared near the top of `main.m` so that the training/test split, cross-validation, visualization sampling, and neural-network validation split can be reproduced.

## Results

<!-- Replace the placeholders below after running the finalized analysis. -->

| Model | Test Accuracy | Macro F1 Score | Notes |
| --- | --- | --- | --- |
| Support Vector Machine | **[add result]** | **[add result]** | **[add observation]** |
| Random Forest | **[add result]** | **[add result]** | **[add observation]** |
| K-Nearest Neighbors | **[add result]** | **[add result]** | **[add observation]** |
| Naive Bayes | **[add result]** | **[add result]** | **[add observation]** |
| Feedforward Neural Network | **[add result]** | **[add result]** | **[add observation]** |

**Key takeaway:** **[Summarize your most important finding here.]**

<!-- Add selected figures below, for example a confusion matrix or training curve. -->

## Validation and Limitations

This is an exploratory educational project, not a clinical diagnostic system. Evaluation choices, possible sources of bias, and interpretation limits should be considered alongside model metrics. See [Analysis.txt](Analysis.txt) for the current analysis notes.

**Validation approach:** **[Describe the final train/test split and cross-validation approach.]**

**Known limitations:** **[Add the limitations and planned improvements from your analysis.]**

## Future Improvements

- **[Add a planned technical improvement.]**
- **[Add a planned validation or data improvement.]**
- **[Add a planned presentation or documentation improvement.]**

## Attribution

The project was inspired by MathWorks' [Wavelet Time Scattering for ECG Signal Classification](https://www.mathworks.com/help/wavelet/ug/ecg-signal-classification-using-wavelet-time-scattering.html) example. The ECG source data is obtained from PhysioNet; dataset-specific citations and license information should be retained with the data setup materials.

## Author

**[Your Name]**  
Student portfolio project  
**[Add your program, school, LinkedIn, or portfolio link]**

## License

**[Choose and add a license for your original code before publishing.]**
