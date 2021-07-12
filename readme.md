# Masked features of task states found in individual brain networks

Here we trained classifiers using a single persons multisession data and tested the model along independent data from the same person or new people. We trained classifiers using all tasks vs rest (binary), all tasks and rest (multiclass), and along single tasks (binary). All machine learning analyses were run through python located in /code/pyScripts/Predictions. To execute all analyses open jupyter notebook and run allPredictions.ipynb. All predictions will be stored as a csv in /output/results/ In order to run these scripts you will need to have [scikit learn](https://scikit-learn.org/stable/install.html) installed. For creation of figures see /code/pyScripts/Figures /FinalFigures.ipynb

## Individualized classifiers can predict task state within and between individuals

All of the main analyses for training along a single subjects data are located in /code/Predictions/masterClass.py This file includes functions for training all vs rest binary, multi-class, single task, and network level. These functions give you the option to select a classifier from sklearn (Log, SVM, Ridge).

## Task state can be decoded from single networks

For feature weight analysis see featureWeights.py for analyzing across folds input subject in function allFolds(). allTasks will output csv files of average feature weights across all folds for each subject for the all task vs rest binary anlaysis. allSubs creates csv's for each subject/task feature weights. To map values onto the brain you have to assign data to parcels located /code/matScripts/assign_data_to_parcel_cifti_V2.m to create all run parcel_batch.m It should be noted for all additional analyses are default Ridge.

To evaluate block level subset of features for all vs rest (binary) and single task are located in block_featSelection.py This function will subset each block of network to network connections. To evaluate each row of networks run runScript located in featSelection.py This script will subset per rows of all of network X to all regions. This script will run for the single task and all task vs rest (binary) analyses. To run random feature selection located in masterClass_rdmNet.py this script will create a log distribution of 40 values that range from 10-50,000 each value is then randomly indexed out of all possible regions. This process of randomly indexing repeats 100 times. Keep in mind this analysis is time exhaustive and may be further improved through other methods.

## Permutation testing

For permutation testing for each analysis run functions located in masterClass_permutation.py This will randomly permute training labels for each analysis repeated the process 1,000 times per analysis.

Permutation testing of the magnitude effect between testing within subject compared to between run scripts in permutGroupLevel.py This will take the distribution of accuracy measures for within and between subject and randomly permute between the two columns. The script then takes the average accuracy for each permuted column and takes the difference of within and between subject creating a null distribution of all permuted difference scores. These values are then compared to the true difference score. P values will display after running each function

## Data quantity

To randomly selection sample pairs between 16-80 run script in manDays.py

## Groupwise approaches

Evaluating how accuracy measures compare to standard leave one subject out cross validation run function in group_approach.py This script will calculate accuracy measures for the all vs rest (binary) and single task analysis. This function stores all subject sessions as a dataframe and selects a single session of data for each subject and runs a standard leave one subject out classification scheme to predict task from rest. 
