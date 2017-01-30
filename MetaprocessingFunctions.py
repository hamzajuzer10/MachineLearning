#!/usr/bin/python

#Prepocess L1 predicted labels

import numpy as np
import sys
import constants
from PreprocessingFunctions import label_transform, label_inv_transform, nonTextFeature_transform, nonTextFeature_nvalues, checkLabelsNFeatures
from sklearn.preprocessing import OneHotEncoder

def predLabelEncoding(predLabels,tag):
    "Encode predicted labels"

    lookup = {'C-FinLossPred':0,
              'C-CusDPred':1,
              'C-RepLossPred':2,
              'C-RegLossPred':3,
              'I-FinLossPred':4,
              'I-CusDPred':5,
              'I-RepLossPred':6,
              'I-RegLoss':7,
              'A-FinLossPred':8,
              'A-CusDPred':9,
              'A-RepLossPred':10,
              'A-RegLossPred':11}

    # determine no. of unique values for each predicted label
    predLabel_nvalues = nonTextFeature_nvalues(constants.nMetaMetaNonText_feature[tag])

    #combine predicted values into feature matrix
    feature_matrix =


    # one-hot encoding
    meta_enc = OneHotEncoder(n_values=predLabel_nvalues)

    # fit train values
    meta_enc.fit(FLfitConfFeaturesNonTextTrain)


    for feature in constants.nMetaMetaNonText_feature[tag]:

        # determine no. of unique values for each feature

    # for each data category clf,
    # for each relevant meta non-text feature
    # encode meta non-text feature (this will be predicted labels for training set)
    # add encoded meta non-text feature to MetaFeatures non text train
    #