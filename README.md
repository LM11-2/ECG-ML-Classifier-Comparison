# ECG Signal Classification with Wavelet Time Scattering

> My personal project comparing machine-learning classifiers for electrocardiogram (ECG) signal classification.

## Project Overview

This project compares several machine-learning classifiers for distinguishing ECG signals across three classes:

- **ARR** — Arrhythmia
- **CHF** — Congestive Heart Failure
- **NSR** — Normal Sinus Rhythm

The pipeline preprocesses raw ECG recordings, extracts features with a wavelet time scattering transform, trains multiple classifiers, and compares their performance. The models included are a support vector machine, random forest, k-nearest neighbors, Naive Bayes classifier, and feedforward neural network, and further information can be found in the [Analysis.md](Analysis.md).

This my independent student portfolio project that spun out of MathWorks' [Wavelet Time Scattering for ECG Signal Classification](https://www.mathworks.com/help/wavelet/ug/ecg-signal-classification-using-wavelet-time-scattering.html) example.

## Project Goals

- Build an end-to-end ECG classification workflow in MATLAB.
- Compare traditional machine-learning models and feedforward neural network.
- Explore wavelet time scattering as an ECG feature-extraction method.
- Improve my MATLAB proficiency

## Workflow

1. Select folder containing the raw ECG recordings (see sourcing information below).
2. Preprocess recordings to a common sampling rate and length.
3. Split the recordings into training and test sets.
4. Extract wavelet time scattering features.
5. Train and evaluate each classifier.
6. Aggregate window-level predictions into record-level predictions using majority vote.
7. Observe and analyze results with summary tables, confusion matrices, and neural-network training curves.

## Repository Guide

| File | Purpose |
| --- | --- |
| `main.m` | Main entry point. Selects the raw-signal folder, runs preprocessing, extracts features, trains models, and displays results. Run this file to observe the comprehensive project.|
| `ECGPreprocessor.m` | Class for loading and standardizing ECG recordings from the selected data folder. |
| `SVMClassifier.m` | Trains and evaluates the support vector machine classifier. |
| `RandForestClassifier.m` | Trains and evaluates the random forest classifier. |
| `KNNClassifier.m` | Trains and evaluates the k-Nearest Neighbors classifier. |
| `NaiveBayesClassifier.m` | Trains and evaluates the Naive Bayes classifier. |
| `NeuralNetClassifier4.m` | Trains and evaluates the feedforward neural network. |
| `randomSplit.m` | Creates a reproducible training/test split. |
| `createSequenceLabels.m` | Matches labels to wavelet-scattering time windows. |
| `helperMajorityVote.m` | Combines window-level predictions into one prediction per ECG record. |
| `displayResults.m` | Ranks and prints classifier results. |
| `plotConfusionMatrices.m` | Creates confusion-matrix visualizations for each model. |
| `plotNNTrainingCurves.m` | Visualizes neural-network training and validation progress. |
| `DownloadInstructions.txt` | Explains how to download and organize the required raw ECG data. |
| `Analysis.md` | Analysis on the project, including validation, limitations and areas of future improvement. |
| `Past Versions/` | Archived project iterations and unused classifier versions. |

## Requirements

This project was developed in MATLAB. Before running it, install or configure:

- MATLAB (created in version R2026a Update 3)
- Wavelet Toolbox
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox
- Signal Processing Toolbox
- Waveform Database Software Package (WFDB) for MATLAB and Octave
    - Instructions can be found in [DownloadInstructions.txt](DownloadInstructions.txt)

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
2. Install the required MATLAB toolboxes.
3. Open MATLAB in the project folder.
4. Run `main.m`.
5. Select the parent folder containing the raw ECG data when prompted.
6. Review the printed model summary and visualizations.

A random seed is declared near the top of `main.m` so that the training/test split, cross-validation, visualization sampling, and neural-network validation split can be reproduced.

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



## Attribution

The project was inspired by MathWorks' [Wavelet Time Scattering for ECG Signal Classification](https://www.mathworks.com/help/wavelet/ug/ecg-signal-classification-using-wavelet-time-scattering.html) example. The ECG source data is obtained from PhysioNet; dataset-specific citations and license information should be retained with the data setup materials.

### Software Citation

This project uses the Waveform Database Software Package (WFDB) for MATLAB and Octave to read WFDB-formatted ECG recordings through `rdsamp`.

Silva, I., Moody, B., & Moody, G. (2021). *Waveform Database Software Package (WFDB) for MATLAB and Octave* (version 0.10.0). PhysioNet. RRID:SCR_007345. [https://doi.org/10.13026/6zcz-e163](https://doi.org/10.13026/6zcz-e163)

## Author

**Luke Mehta**  
*BME + CS, Johns Hopkins University*

## License

This project's original code is released under the MIT License. The external datasets and WFDB software retain their own licenses, citation requirements, and terms of use.
