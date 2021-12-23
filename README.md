# Gene Resistance Classification

## Problem
Given the dataset `dataset.npy`, make a 80%-20% train-test split. Train a deep neural network that is based on RNN, CNN, or both. Evaluate the model on the test set.

A data point in the dataset consists of a gene and its resistance status against a drug. A gene is a string of DNA letters that gets translated into a protein, or a string of amino acid.

## Hints
- You may translate the genes (DNA strings) into proteins (amino acid strings), although it's not necessary to do so to achieve high performance. You can find a map for how to convert triplets of DNA to protein letters in the accompanying file codons_aa.txt
- Depending on the availability of comptutational resource, you may downsample the number of training examples.