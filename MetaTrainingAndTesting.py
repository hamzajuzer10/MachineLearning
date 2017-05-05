#!/usr/bin/python

#Train machine learning algorithms on features and labels

import numpy as np
import constants
import imp
import sys
import os
import datetime
from time import time
import FeatureTrainingExtraction
from sklearn.metrics import accuracy_score
from PreprocessingFunctions import label_transform, metaNonTextFeature_combine, nonTextFeature_transform, nonTextFeature_nvalues, checkLabelsNFeatures



def MetaTrainAndTest(mconf_pred, mconf_trpred, core_directory):

    # Labels
    print 'transforming meta training and test labels...'
    # train and test labels
    MetatransLabelsTrain = label_transform(FeatureTrainingExtraction.trainingLabels)
    MetatransLabelsTest = label_transform(FeatureTrainingExtraction.testLabels)

    # Features (non-text)


    print 'transforming meta training and test features (non-text)...'
    # train
    MetaFLfitConfFeaturesNonTextTrain = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaFLconfidentialityNonTextTrainingFeatures)
    MetaCDfitConfFeaturesNonTextTrain = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaCDconfidentialityNonTextTrainingFeatures)
    MetaRPfitConfFeaturesNonTextTrain = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaRPconfidentialityNonTextTrainingFeatures)
    MetaRGfitConfFeaturesNonTextTrain = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaRGconfidentialityNonTextTrainingFeatures)
    MetaFLfitIntFeaturesNonTextTrain = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaFLintegrityNonTextTrainingFeatures)
    MetaCDfitIntFeaturesNonTextTrain = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaCDintegrityNonTextTrainingFeatures)
    MetaRPfitIntFeaturesNonTextTrain = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaRPintegrityNonTextTrainingFeatures)
    MetaRGfitIntFeaturesNonTextTrain = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaRGintegrityNonTextTrainingFeatures)
    MetaFLfitAvailFeaturesNonTextTrain = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaFLavailabilityNonTextTrainingFeatures)
    MetaCDfitAvailFeaturesNonTextTrain = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaCDavailabilityNonTextTrainingFeatures)
    MetaRPfitAvailFeaturesNonTextTrain = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaRPavailabilityNonTextTrainingFeatures)
    MetaRGfitAvailFeaturesNonTextTrain = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaRGavailabilityNonTextTrainingFeatures)

    # test
    MetaFLfitConfFeaturesNonTextTest = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaFLconfidentialityNonTextTestFeatures)
    MetaCDfitConfFeaturesNonTextTest = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaCDconfidentialityNonTextTestFeatures)
    MetaRPfitConfFeaturesNonTextTest = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaRPconfidentialityNonTextTestFeatures)
    MetaRGfitConfFeaturesNonTextTest = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaRGconfidentialityNonTextTestFeatures)
    MetaFLfitIntFeaturesNonTextTest = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaFLintegrityNonTextTestFeatures)
    MetaCDfitIntFeaturesNonTextTest = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaCDintegrityNonTextTestFeatures)
    MetaRPfitIntFeaturesNonTextTest = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaRPintegrityNonTextTestFeatures)
    MetaRGfitIntFeaturesNonTextTest = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaRGintegrityNonTextTestFeatures)
    MetaFLfitAvailFeaturesNonTextTest = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaFLavailabilityNonTextTestFeatures)
    MetaCDfitAvailFeaturesNonTextTest = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaCDavailabilityNonTextTestFeatures)
    MetaRPfitAvailFeaturesNonTextTest = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaRPavailabilityNonTextTestFeatures)
    MetaRGfitAvailFeaturesNonTextTest = nonTextFeature_transform(
        FeatureTrainingExtraction.MetaRGavailabilityNonTextTestFeatures)

    # add predicted test and train labels as features
    # train
    MetaFLfitConfFeaturesNonTextTrain = metaNonTextFeature_combine(MetaFLfitConfFeaturesNonTextTrain, mconf_trpred,
                                                                   'FLconfidentialityNonText_features')
    MetaCDfitConfFeaturesNonTextTrain = metaNonTextFeature_combine(MetaCDfitConfFeaturesNonTextTrain, mconf_trpred,
                                                                   'CDconfidentialityNonText_features')
    MetaRPfitConfFeaturesNonTextTrain = metaNonTextFeature_combine(MetaRPfitConfFeaturesNonTextTrain, mconf_trpred,
                                                                   'RPconfidentialityNonText_features')
    MetaRGfitConfFeaturesNonTextTrain = metaNonTextFeature_combine(MetaRGfitConfFeaturesNonTextTrain, mconf_trpred,
                                                                   'RGconfidentialityNonText_features')
    MetaFLfitIntFeaturesNonTextTrain = metaNonTextFeature_combine(MetaFLfitIntFeaturesNonTextTrain, mconf_trpred,
                                                                  'FLintegrityNonText_features')
    MetaCDfitIntFeaturesNonTextTrain = metaNonTextFeature_combine(MetaCDfitIntFeaturesNonTextTrain, mconf_trpred,
                                                                  'CDintegrityNonText_features')
    MetaRPfitIntFeaturesNonTextTrain = metaNonTextFeature_combine(MetaRPfitIntFeaturesNonTextTrain, mconf_trpred,
                                                                  'RPintegrityNonText_features')
    MetaRGfitIntFeaturesNonTextTrain = metaNonTextFeature_combine(MetaRGfitIntFeaturesNonTextTrain, mconf_trpred,
                                                                  'RGintegrityNonText_features')
    MetaFLfitAvailFeaturesNonTextTrain = metaNonTextFeature_combine(MetaFLfitAvailFeaturesNonTextTrain, mconf_trpred,
                                                                    'FLavailabilityNonText_features')
    MetaCDfitAvailFeaturesNonTextTrain = metaNonTextFeature_combine(MetaCDfitAvailFeaturesNonTextTrain, mconf_trpred,
                                                                    'CDavailabilityNonText_features')
    MetaRPfitAvailFeaturesNonTextTrain = metaNonTextFeature_combine(MetaRPfitAvailFeaturesNonTextTrain, mconf_trpred,
                                                                    'RPavailabilityNonText_features')
    MetaRGfitAvailFeaturesNonTextTrain = metaNonTextFeature_combine(MetaRGfitAvailFeaturesNonTextTrain, mconf_trpred,
                                                                    'RGavailabilityNonText_features')

    # test

    MetaFLfitConfFeaturesNonTextTest = metaNonTextFeature_combine(MetaFLfitConfFeaturesNonTextTest, mconf_pred,
                                                                  'FLconfidentialityNonText_features')
    MetaCDfitConfFeaturesNonTextTest = metaNonTextFeature_combine(MetaCDfitConfFeaturesNonTextTest, mconf_pred,
                                                                  'CDconfidentialityNonText_features')
    MetaRPfitConfFeaturesNonTextTest = metaNonTextFeature_combine(MetaRPfitConfFeaturesNonTextTest, mconf_pred,
                                                                  'RPconfidentialityNonText_features')
    MetaRGfitConfFeaturesNonTextTest = metaNonTextFeature_combine(MetaRGfitConfFeaturesNonTextTest, mconf_pred,
                                                                  'RGconfidentialityNonText_features')
    MetaFLfitIntFeaturesNonTextTest = metaNonTextFeature_combine(MetaFLfitIntFeaturesNonTextTest, mconf_pred,
                                                                 'FLintegrityNonText_features')
    MetaCDfitIntFeaturesNonTextTest = metaNonTextFeature_combine(MetaCDfitIntFeaturesNonTextTest, mconf_pred,
                                                                 'CDintegrityNonText_features')
    MetaRPfitIntFeaturesNonTextTest = metaNonTextFeature_combine(MetaRPfitIntFeaturesNonTextTest, mconf_pred,
                                                                 'RPintegrityNonText_features')
    MetaRGfitIntFeaturesNonTextTest = metaNonTextFeature_combine(MetaRGfitIntFeaturesNonTextTest, mconf_pred,
                                                                 'RGintegrityNonText_features')
    MetaFLfitAvailFeaturesNonTextTest = metaNonTextFeature_combine(MetaFLfitAvailFeaturesNonTextTest, mconf_pred,
                                                                   'FLavailabilityNonText_features')
    MetaCDfitAvailFeaturesNonTextTest = metaNonTextFeature_combine(MetaCDfitAvailFeaturesNonTextTest, mconf_pred,
                                                                   'CDavailabilityNonText_features')
    MetaRPfitAvailFeaturesNonTextTest = metaNonTextFeature_combine(MetaRPfitAvailFeaturesNonTextTest, mconf_pred,
                                                                   'RPavailabilityNonText_features')
    MetaRGfitAvailFeaturesNonTextTest = metaNonTextFeature_combine(MetaRGfitAvailFeaturesNonTextTest, mconf_pred,
                                                                   'RGavailabilityNonText_features')

    # determine no. of unique values for each feature

    # combine the Base and Meta features
    MetaFLconf_features = constants.nMetaBaseNonText_feature['FLconfidentialityNonText_features'] + \
                          constants.nMetaMetaNonText_feature['FLconfidentialityNonText_features']
    MetaCDconf_features = constants.nMetaBaseNonText_feature['CDconfidentialityNonText_features'] + \
                          constants.nMetaMetaNonText_feature['CDconfidentialityNonText_features']
    MetaRPconf_features = constants.nMetaBaseNonText_feature['RPconfidentialityNonText_features'] + \
                          constants.nMetaMetaNonText_feature['RPconfidentialityNonText_features']
    MetaRGconf_features = constants.nMetaBaseNonText_feature['RGconfidentialityNonText_features'] + \
                          constants.nMetaMetaNonText_feature['RGconfidentialityNonText_features']
    MetaFLint_features = constants.nMetaBaseNonText_feature['FLintegrityNonText_features'] + \
                         constants.nMetaMetaNonText_feature['FLintegrityNonText_features']
    MetaCDint_features = constants.nMetaBaseNonText_feature['CDintegrityNonText_features'] + \
                         constants.nMetaMetaNonText_feature['CDintegrityNonText_features']
    MetaRPint_features = constants.nMetaBaseNonText_feature['RPintegrityNonText_features'] + \
                         constants.nMetaMetaNonText_feature['RPintegrityNonText_features']
    MetaRGint_features = constants.nMetaBaseNonText_feature['RGintegrityNonText_features'] + \
                         constants.nMetaMetaNonText_feature['RGintegrityNonText_features']
    MetaFLavail_features = constants.nMetaBaseNonText_feature['FLavailabilityNonText_features'] + \
                           constants.nMetaMetaNonText_feature['FLavailabilityNonText_features']
    MetaCDavail_features = constants.nMetaBaseNonText_feature['CDavailabilityNonText_features'] + \
                           constants.nMetaMetaNonText_feature['CDavailabilityNonText_features']
    MetaRPavail_features = constants.nMetaBaseNonText_feature['RPavailabilityNonText_features'] + \
                           constants.nMetaMetaNonText_feature['RPavailabilityNonText_features']
    MetaRGavail_features = constants.nMetaBaseNonText_feature['RGavailabilityNonText_features'] + \
                           constants.nMetaMetaNonText_feature['RGavailabilityNonText_features']

    # determine no. of unique values for each feature
    MetaFLconf_nvalues = nonTextFeature_nvalues(MetaFLconf_features)
    MetaCDconf_nvalues = nonTextFeature_nvalues(MetaCDconf_features)
    MetaRPconf_nvalues = nonTextFeature_nvalues(MetaRPconf_features)
    MetaRGconf_nvalues = nonTextFeature_nvalues(MetaRGconf_features)
    MetaFLint_nvalues = nonTextFeature_nvalues(MetaFLint_features)
    MetaCDint_nvalues = nonTextFeature_nvalues(MetaCDint_features)
    MetaRPint_nvalues = nonTextFeature_nvalues(MetaRPint_features)
    MetaRGint_nvalues = nonTextFeature_nvalues(MetaRGint_features)
    MetaFLavail_nvalues = nonTextFeature_nvalues(MetaFLavail_features)
    MetaCDavail_nvalues = nonTextFeature_nvalues(MetaCDavail_features)
    MetaRPavail_nvalues = nonTextFeature_nvalues(MetaRPavail_features)
    MetaRGavail_nvalues = nonTextFeature_nvalues(MetaRGavail_features)

    from sklearn.preprocessing import OneHotEncoder
    # one-hot encoding
    MetaFLconf_enc = OneHotEncoder(n_values=MetaFLconf_nvalues)
    MetaCDconf_enc = OneHotEncoder(n_values=MetaCDconf_nvalues)
    MetaRPconf_enc = OneHotEncoder(n_values=MetaRPconf_nvalues)
    MetaRGconf_enc = OneHotEncoder(n_values=MetaRGconf_nvalues)
    MetaFLint_enc = OneHotEncoder(n_values=MetaFLint_nvalues)
    MetaCDint_enc = OneHotEncoder(n_values=MetaCDint_nvalues)
    MetaRPint_enc = OneHotEncoder(n_values=MetaRPint_nvalues)
    MetaRGint_enc = OneHotEncoder(n_values=MetaRGint_nvalues)
    MetaFLavail_enc = OneHotEncoder(n_values=MetaFLavail_nvalues)
    MetaCDavail_enc = OneHotEncoder(n_values=MetaCDavail_nvalues)
    MetaRPavail_enc = OneHotEncoder(n_values=MetaRPavail_nvalues)
    MetaRGavail_enc = OneHotEncoder(n_values=MetaRGavail_nvalues)

    print 'preprocessing meta training and test features (non-text)...'
    # fit train values
    MetaFLconf_enc.fit(MetaFLfitConfFeaturesNonTextTrain)
    MetaCDconf_enc.fit(MetaCDfitConfFeaturesNonTextTrain)
    MetaRPconf_enc.fit(MetaRPfitConfFeaturesNonTextTrain)
    MetaRGconf_enc.fit(MetaRGfitConfFeaturesNonTextTrain)
    MetaFLint_enc.fit(MetaFLfitIntFeaturesNonTextTrain)
    MetaCDint_enc.fit(MetaCDfitIntFeaturesNonTextTrain)
    MetaRPint_enc.fit(MetaRPfitIntFeaturesNonTextTrain)
    MetaRGint_enc.fit(MetaRGfitIntFeaturesNonTextTrain)
    MetaFLavail_enc.fit(MetaFLfitAvailFeaturesNonTextTrain)
    MetaCDavail_enc.fit(MetaCDfitAvailFeaturesNonTextTrain)
    MetaRPavail_enc.fit(MetaRPfitAvailFeaturesNonTextTrain)
    MetaRGavail_enc.fit(MetaRGfitAvailFeaturesNonTextTrain)

    # transform train values
    MetaFLtransConfFeaturesNonTextTrain = MetaFLconf_enc.transform(MetaFLfitConfFeaturesNonTextTrain).toarray()
    MetaCDtransConfFeaturesNonTextTrain = MetaCDconf_enc.transform(MetaCDfitConfFeaturesNonTextTrain).toarray()
    MetaRPtransConfFeaturesNonTextTrain = MetaRPconf_enc.transform(MetaRPfitConfFeaturesNonTextTrain).toarray()
    MetaRGtransConfFeaturesNonTextTrain = MetaRGconf_enc.transform(MetaRGfitConfFeaturesNonTextTrain).toarray()
    MetaFLtransIntFeaturesNonTextTrain = MetaFLint_enc.transform(MetaFLfitIntFeaturesNonTextTrain).toarray()
    MetaCDtransIntFeaturesNonTextTrain = MetaCDint_enc.transform(MetaCDfitIntFeaturesNonTextTrain).toarray()
    MetaRPtransIntFeaturesNonTextTrain = MetaRPint_enc.transform(MetaRPfitIntFeaturesNonTextTrain).toarray()
    MetaRGtransIntFeaturesNonTextTrain = MetaRGint_enc.transform(MetaRGfitIntFeaturesNonTextTrain).toarray()
    MetaFLtransAvailFeaturesNonTextTrain = MetaFLavail_enc.transform(MetaFLfitAvailFeaturesNonTextTrain).toarray()
    MetaCDtransAvailFeaturesNonTextTrain = MetaCDavail_enc.transform(MetaCDfitAvailFeaturesNonTextTrain).toarray()
    MetaRPtransAvailFeaturesNonTextTrain = MetaRPavail_enc.transform(MetaRPfitAvailFeaturesNonTextTrain).toarray()
    MetaRGtransAvailFeaturesNonTextTrain = MetaRGavail_enc.transform(MetaRGfitAvailFeaturesNonTextTrain).toarray()

    # transform test values
    if constants.train_test_split != 1.0:
        MetaFLtransConfFeaturesNonTextTest = MetaFLconf_enc.transform(MetaFLfitConfFeaturesNonTextTest).toarray()
        MetaCDtransConfFeaturesNonTextTest = MetaCDconf_enc.transform(MetaCDfitConfFeaturesNonTextTest).toarray()
        MetaRPtransConfFeaturesNonTextTest = MetaRPconf_enc.transform(MetaRPfitConfFeaturesNonTextTest).toarray()
        MetaRGtransConfFeaturesNonTextTest = MetaRGconf_enc.transform(MetaRGfitConfFeaturesNonTextTest).toarray()
        MetaFLtransIntFeaturesNonTextTest = MetaFLint_enc.transform(MetaFLfitIntFeaturesNonTextTest).toarray()
        MetaCDtransIntFeaturesNonTextTest = MetaCDint_enc.transform(MetaCDfitIntFeaturesNonTextTest).toarray()
        MetaRPtransIntFeaturesNonTextTest = MetaRPint_enc.transform(MetaRPfitIntFeaturesNonTextTest).toarray()
        MetaRGtransIntFeaturesNonTextTest = MetaRGint_enc.transform(MetaRGfitIntFeaturesNonTextTest).toarray()
        MetaFLtransAvailFeaturesNonTextTest = MetaFLavail_enc.transform(MetaFLfitAvailFeaturesNonTextTest).toarray()
        MetaCDtransAvailFeaturesNonTextTest = MetaCDavail_enc.transform(MetaCDfitAvailFeaturesNonTextTest).toarray()
        MetaRPtransAvailFeaturesNonTextTest = MetaRPavail_enc.transform(MetaRPfitAvailFeaturesNonTextTest).toarray()
        MetaRGtransAvailFeaturesNonTextTest = MetaRGavail_enc.transform(MetaRGfitAvailFeaturesNonTextTest).toarray()

    # #feature selection, to reduce overall dimensionality and computational time - TODO


    # verify correct labels and features
    print 'verifying accuracy of labels and feature matrices...'
    checkLabelsNFeatures(MetaFLtransConfFeaturesNonTextTrain, MetatransLabelsTrain[0], 'FL confidentiality training')
    checkLabelsNFeatures(MetaCDtransConfFeaturesNonTextTrain, MetatransLabelsTrain[1], 'CD confidentiality training')
    checkLabelsNFeatures(MetaRPtransConfFeaturesNonTextTrain, MetatransLabelsTrain[2], 'RP confidentiality training')
    checkLabelsNFeatures(MetaRGtransConfFeaturesNonTextTrain, MetatransLabelsTrain[3], 'RG confidentiality training')
    checkLabelsNFeatures(MetaFLtransIntFeaturesNonTextTrain, MetatransLabelsTrain[4], 'FL integrity training')
    checkLabelsNFeatures(MetaCDtransIntFeaturesNonTextTrain, MetatransLabelsTrain[5], 'CD integrity training')
    checkLabelsNFeatures(MetaRPtransIntFeaturesNonTextTrain, MetatransLabelsTrain[6], 'RP integrity training')
    checkLabelsNFeatures(MetaRGtransIntFeaturesNonTextTrain, MetatransLabelsTrain[7], 'RG integrity training')
    checkLabelsNFeatures(MetaFLtransAvailFeaturesNonTextTrain, MetatransLabelsTrain[8], 'FL availability training')
    checkLabelsNFeatures(MetaCDtransAvailFeaturesNonTextTrain, MetatransLabelsTrain[9], 'CD availability training')
    checkLabelsNFeatures(MetaRPtransAvailFeaturesNonTextTrain, MetatransLabelsTrain[10], 'RP availability training')
    checkLabelsNFeatures(MetaRGtransAvailFeaturesNonTextTrain, MetatransLabelsTrain[11], 'RG availability training')

    if constants.train_test_split != 1.0:
        checkLabelsNFeatures(MetaFLtransConfFeaturesNonTextTest, MetatransLabelsTest[0], 'FL confidentiality testing')
        checkLabelsNFeatures(MetaCDtransConfFeaturesNonTextTest, MetatransLabelsTest[1], 'CD confidentiality testing')
        checkLabelsNFeatures(MetaRPtransConfFeaturesNonTextTest, MetatransLabelsTest[2], 'RP confidentiality testing')
        checkLabelsNFeatures(MetaRGtransConfFeaturesNonTextTest, MetatransLabelsTest[3], 'RG confidentiality testing')
        checkLabelsNFeatures(MetaFLtransIntFeaturesNonTextTest, MetatransLabelsTest[4], 'FL integrity testing')
        checkLabelsNFeatures(MetaCDtransIntFeaturesNonTextTest, MetatransLabelsTest[5], 'CD integrity testing')
        checkLabelsNFeatures(MetaRPtransIntFeaturesNonTextTest, MetatransLabelsTest[6], 'RP integrity testing')
        checkLabelsNFeatures(MetaRGtransIntFeaturesNonTextTest, MetatransLabelsTest[7], 'RG integrity testing')
        checkLabelsNFeatures(MetaFLtransAvailFeaturesNonTextTest, MetatransLabelsTest[8], 'FL availability testing')
        checkLabelsNFeatures(MetaCDtransAvailFeaturesNonTextTest, MetatransLabelsTest[9], 'CD availability testing')
        checkLabelsNFeatures(MetaRPtransAvailFeaturesNonTextTest, MetatransLabelsTest[10], 'RP availability testing')
        checkLabelsNFeatures(MetaRGtransAvailFeaturesNonTextTest, MetatransLabelsTest[11], 'RG availability testing')

    # Run training
    t0 = time()

    print 'training L2 classifier...'

    label_clf = {}

    for label_t,classifier_t in constants.meta_classifier.items():

        from sklearn.grid_search import GridSearchCV
        if classifier_t == 'GaussianNaiveBayes':
            from sklearn.naive_bayes import GaussianNB
            label_clf[label_t] = GridSearchCV(GaussianNB(),constants.meta_param_grid[label_t])

        elif classifier_t == 'DecisionTree':
            from sklearn.tree import DecisionTreeClassifier
            label_clf[label_t] = GridSearchCV(DecisionTreeClassifier(),constants.meta_param_grid[label_t])

        elif classifier_t == 'SVM':
            from sklearn.svm import SVC
            label_clf[label_t] = GridSearchCV(SVC(),constants.meta_param_grid[label_t])

        elif classifier_t == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            label_clf[label_t] = GridSearchCV(KNeighborsClassifier(),constants.meta_param_grid[label_t])

        elif classifier_t == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier
            label_clf[label_t] = GridSearchCV(RandomForestClassifier(),constants.meta_param_grid[label_t])

        elif classifier_t == 'AdaBoostDT':
            from sklearn.ensemble import AdaBoostClassifier
            from sklearn.tree import DecisionTreeClassifier
            label_clf[label_t] = GridSearchCV(AdaBoostClassifier(base_estimator = DecisionTreeClassifier()),constants.meta_param_grid[label_t])

        else:
            print 'no classifier selected...'
            print 'program is exiting...'
            sys.exit(0)

    #fit classifier
    MetaFLconf_clf = label_clf['FLconfidentiality']
    MetaCDconf_clf = label_clf['CDconfidentiality']
    MetaRPconf_clf = label_clf['RPconfidentiality']
    MetaRGconf_clf = label_clf['RGconfidentiality']
    MetaFLint_clf = label_clf['FLintegrity']
    MetaCDint_clf = label_clf['CDintegrity']
    MetaRPint_clf = label_clf['RPintegrity']
    MetaRGint_clf = label_clf['RGintegrity']
    MetaFLavail_clf = label_clf['FLavailability']
    MetaCDavail_clf = label_clf['CDavailability']
    MetaRPavail_clf = label_clf['RPavailability']
    MetaRGavail_clf = label_clf['RGavailability']


    print 'training %s L2 classifier on Confidentiality FinLoss...' % constants.meta_classifier['FLconfidentiality']
    MetaFLconf_clf.fit(MetaFLtransConfFeaturesNonTextTrain,MetatransLabelsTrain[0])

    print 'training %s L2 classifier on Confidentiality CusD...' % constants.meta_classifier['CDconfidentiality']
    MetaCDconf_clf.fit(MetaCDtransConfFeaturesNonTextTrain,MetatransLabelsTrain[1])

    print 'training %s L2 classifier on Confidentiality RepLoss...' % constants.meta_classifier['RPconfidentiality']
    MetaRPconf_clf.fit(MetaRPtransConfFeaturesNonTextTrain,MetatransLabelsTrain[2])

    print 'training %s L2 classifier on Confidentiality RegLoss...' % constants.meta_classifier['RGconfidentiality']
    MetaRGconf_clf.fit(MetaRGtransConfFeaturesNonTextTrain,MetatransLabelsTrain[3])

    print 'training %s L2 classifier on Integrity FinLoss...' % constants.meta_classifier['FLintegrity']
    MetaFLint_clf.fit(MetaFLtransIntFeaturesNonTextTrain,MetatransLabelsTrain[4])

    print 'training %s L2 classifier on Integrity CusD...' % constants.meta_classifier['CDintegrity']
    MetaCDint_clf.fit(MetaCDtransIntFeaturesNonTextTrain,MetatransLabelsTrain[5])

    print 'training %s L2 classifier on Integrity RepLoss...' % constants.meta_classifier['RPintegrity']
    MetaRPint_clf.fit(MetaRPtransIntFeaturesNonTextTrain,MetatransLabelsTrain[6])

    print 'training %s L2 classifier on Integrity RegLoss...' % constants.meta_classifier['RGintegrity']
    MetaRGint_clf.fit(MetaRGtransIntFeaturesNonTextTrain,MetatransLabelsTrain[7])

    print 'training %s L2 classifier on Availability FinLoss...' % constants.meta_classifier['FLavailability']
    MetaFLavail_clf.fit(MetaFLtransAvailFeaturesNonTextTrain,MetatransLabelsTrain[8])

    print 'training %s L2 classifier on Availability CusD...' % constants.meta_classifier['CDavailability']
    MetaCDavail_clf.fit(MetaCDtransAvailFeaturesNonTextTrain,MetatransLabelsTrain[9])

    print 'training %s L2 classifier on Availability RepLoss...' % constants.meta_classifier['RPavailability']
    MetaRPavail_clf.fit(MetaRPtransAvailFeaturesNonTextTrain,MetatransLabelsTrain[10])

    print 'training %s L2 classifier on Availability RegLoss...' % constants.meta_classifier['RGavailability']
    MetaRGavail_clf.fit(MetaRGtransAvailFeaturesNonTextTrain,MetatransLabelsTrain[11])



    nb_training_time = round(time()-t0,3)

    print 'L2 training complete: ',nb_training_time, 's'
    print 'computing test and train prediction results for L2 classifier...'

    if constants.train_test_split != 1.0:

        print 'running test prediction on L2 classifier...'

        #compute test results
        MetaFLconf_pred = MetaFLconf_clf.predict(MetaFLtransConfFeaturesNonTextTest)
        MetaCDconf_pred = MetaCDconf_clf.predict(MetaCDtransConfFeaturesNonTextTest)
        MetaRPconf_pred = MetaRPconf_clf.predict(MetaRPtransConfFeaturesNonTextTest)
        MetaRGconf_pred = MetaRGconf_clf.predict(MetaRGtransConfFeaturesNonTextTest)
        MetaFLint_pred = MetaFLint_clf.predict(MetaFLtransIntFeaturesNonTextTest)
        MetaCDint_pred = MetaCDint_clf.predict(MetaCDtransIntFeaturesNonTextTest)
        MetaRPint_pred = MetaRPint_clf.predict(MetaRPtransIntFeaturesNonTextTest)
        MetaRGint_pred = MetaRGint_clf.predict(MetaRGtransIntFeaturesNonTextTest)
        MetaFLavail_pred = MetaFLavail_clf.predict(MetaFLtransAvailFeaturesNonTextTest)
        MetaCDavail_pred = MetaCDavail_clf.predict(MetaCDtransAvailFeaturesNonTextTest)
        MetaRPavail_pred = MetaRPavail_clf.predict(MetaRPtransAvailFeaturesNonTextTest)
        MetaRGavail_pred = MetaRGavail_clf.predict(MetaRGtransAvailFeaturesNonTextTest)


        print 'test prediction complete...'

        #compute test accuracy
        classifier_acc=[]
        classifier_acc.append(accuracy_score(MetaFLconf_pred, MetatransLabelsTest[0]))
        classifier_acc.append(accuracy_score(MetaCDconf_pred, MetatransLabelsTest[1]))
        classifier_acc.append(accuracy_score(MetaRPconf_pred, MetatransLabelsTest[2]))
        classifier_acc.append(accuracy_score(MetaRGconf_pred, MetatransLabelsTest[3]))
        classifier_acc.append(accuracy_score(MetaFLint_pred, MetatransLabelsTest[4]))
        classifier_acc.append(accuracy_score(MetaCDint_pred, MetatransLabelsTest[5]))
        classifier_acc.append(accuracy_score(MetaRPint_pred, MetatransLabelsTest[6]))
        classifier_acc.append(accuracy_score(MetaRGint_pred, MetatransLabelsTest[7]))
        classifier_acc.append(accuracy_score(MetaFLavail_pred, MetatransLabelsTest[8]))
        classifier_acc.append(accuracy_score(MetaCDavail_pred, MetatransLabelsTest[9]))
        classifier_acc.append(accuracy_score(MetaRPavail_pred, MetatransLabelsTest[10]))
        classifier_acc.append(accuracy_score(MetaRGavail_pred, MetatransLabelsTest[11]))

        print 'Calculating prediction accuracy for test set...'
        print 'FL confidentiality prediction accuracy: ', classifier_acc[0]
        print 'CD confidentiality prediction accuracy: ', classifier_acc[1]
        print 'RP confidentiality prediction accuracy: ', classifier_acc[2]
        print 'RG confidentiality prediction accuracy: ', classifier_acc[3]
        print 'FL integrity prediction accuracy: ', classifier_acc[4]
        print 'CD integrity prediction accuracy: ', classifier_acc[5]
        print 'RP integrity prediction accuracy: ', classifier_acc[6]
        print 'RG integrity prediction accuracy: ', classifier_acc[7]
        print 'FL availability prediction accuracy: ', classifier_acc[8]
        print 'CD availability prediction accuracy: ', classifier_acc[9]
        print 'RP availability prediction accuracy: ', classifier_acc[10]
        print 'RG availability prediction accuracy: ', classifier_acc[11]
        print 'Overall test accuracy: ', np.mean(classifier_acc)


    print 'running training prediction on L1 classifier...'

    # computing train results for bias/variance score
    MetaFLconf_trpred = MetaFLconf_clf.predict(MetaFLtransConfFeaturesNonTextTrain)
    MetaCDconf_trpred = MetaCDconf_clf.predict(MetaCDtransConfFeaturesNonTextTrain)
    MetaRPconf_trpred = MetaRPconf_clf.predict(MetaRPtransConfFeaturesNonTextTrain)
    MetaRGconf_trpred = MetaRGconf_clf.predict(MetaRGtransConfFeaturesNonTextTrain)
    MetaFLint_trpred = MetaFLint_clf.predict(MetaFLtransIntFeaturesNonTextTrain)
    MetaCDint_trpred = MetaCDint_clf.predict(MetaCDtransIntFeaturesNonTextTrain)
    MetaRPint_trpred = MetaRPint_clf.predict(MetaRPtransIntFeaturesNonTextTrain)
    MetaRGint_trpred = MetaRGint_clf.predict(MetaRGtransIntFeaturesNonTextTrain)
    MetaFLavail_trpred = MetaFLavail_clf.predict(MetaFLtransAvailFeaturesNonTextTrain)
    MetaCDavail_trpred = MetaCDavail_clf.predict(MetaCDtransAvailFeaturesNonTextTrain)
    MetaRPavail_trpred = MetaRPavail_clf.predict(MetaRPtransAvailFeaturesNonTextTrain)
    MetaRGavail_trpred = MetaRGavail_clf.predict(MetaRGtransAvailFeaturesNonTextTrain)

    print 'training prediction complete...'

    # compute train accuracy
    classifier_tacc = []
    classifier_tacc.append(accuracy_score(MetaFLconf_trpred, MetatransLabelsTrain[0]))
    classifier_tacc.append(accuracy_score(MetaCDconf_trpred, MetatransLabelsTrain[1]))
    classifier_tacc.append(accuracy_score(MetaRPconf_trpred, MetatransLabelsTrain[2]))
    classifier_tacc.append(accuracy_score(MetaRGconf_trpred, MetatransLabelsTrain[3]))
    classifier_tacc.append(accuracy_score(MetaFLint_trpred, MetatransLabelsTrain[4]))
    classifier_tacc.append(accuracy_score(MetaCDint_trpred, MetatransLabelsTrain[5]))
    classifier_tacc.append(accuracy_score(MetaRPint_trpred, MetatransLabelsTrain[6]))
    classifier_tacc.append(accuracy_score(MetaRGint_trpred, MetatransLabelsTrain[7]))
    classifier_tacc.append(accuracy_score(MetaFLavail_trpred, MetatransLabelsTrain[8]))
    classifier_tacc.append(accuracy_score(MetaCDavail_trpred, MetatransLabelsTrain[9]))
    classifier_tacc.append(accuracy_score(MetaRPavail_trpred, MetatransLabelsTrain[10]))
    classifier_tacc.append(accuracy_score(MetaRGavail_trpred, MetatransLabelsTrain[11]))

    print 'Calculating prediction accuracy for training set...'
    print 'FL confidentiality prediction accuracy: ', classifier_tacc[0]
    print 'CD confidentiality prediction accuracy: ', classifier_tacc[1]
    print 'RP confidentiality prediction accuracy: ', classifier_tacc[2]
    print 'RG confidentiality prediction accuracy: ', classifier_tacc[3]
    print 'FL integrity prediction accuracy: ', classifier_tacc[4]
    print 'CD integrity prediction accuracy: ', classifier_tacc[5]
    print 'RP integrity prediction accuracy: ', classifier_tacc[6]
    print 'RG integrity prediction accuracy: ', classifier_tacc[7]
    print 'FL availability prediction accuracy: ', classifier_tacc[8]
    print 'CD availability prediction accuracy: ', classifier_tacc[9]
    print 'RP availability prediction accuracy: ', classifier_tacc[10]
    print 'RG availability prediction accuracy: ', classifier_tacc[11]
    print 'Overall train accuracy: ', np.mean(classifier_tacc)


    if constants.save:
        print 'saving L2 training classifier data...'

        from sklearn.externals import joblib
        directory = core_directory + '/'
        directory += 'L2_classifier'

        if constants.train_test_split != 1.0:
            directory += '_'
            directory += str(int(np.mean(classifier_acc) * 100))
            directory += '%_acc'

        # make directory if one doesnt exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        #filename (clfs)
        cl_1 = directory + str('/MetaFLconf.pkl')
        cl_2 = directory + str('/MetaCDconf.pkl')
        cl_3 = directory + str('/MetaRPconf.pkl')
        cl_4 = directory + str('/MetaRGconf.pkl')
        cl_5 = directory + str('/MetaFLint.pkl')
        cl_6 = directory + str('/MetaCDint.pkl')
        cl_7 = directory + str('/MetaRPint.pkl')
        cl_8 = directory + str('/MetaRGint.pkl')
        cl_9 = directory + str('/MetaFLavail.pkl')
        cl_10= directory + str('/MetaCDavail.pkl')
        cl_11= directory + str('/MetaRPavail.pkl')
        cl_12= directory + str('/MetaRGavail.pkl')

        #filename (output metrics)
        outputMetrics = directory +str('/Metametrics.txt')

        #filename (features) x2
        tf1 = directory + str('/nMetaBaseNonTextFeature.pkl')
        tf2 = directory + str('/nMetaMetaNonTextFeature.pkl')

        #filename (encoders)
        en_1 = directory + str('/MetaFLconfEncoder.pkl')
        en_2 = directory + str('/MetaCDconfEncoder.pkl')
        en_3 = directory + str('/MetaRPconfEncoder.pkl')
        en_4 = directory + str('/MetaRGconfEncoder.pkl')
        en_5 = directory + str('/MetaFLintEncoder.pkl')
        en_6 = directory + str('/MetaCDintEncoder.pkl')
        en_7 = directory + str('/MetaRPintEncoder.pkl')
        en_8 = directory + str('/MetaRGintEncoder.pkl')
        en_9 = directory + str('/MetaFLavailEncoder.pkl')
        en_10 = directory + str('/MetaCDavailEncoder.pkl')
        en_11 = directory + str('/MetaRPavailEncoder.pkl')
        en_12 = directory + str('/MetaRGavailEncoder.pkl')

        #save clfs
        joblib.dump(MetaFLconf_clf, cl_1)
        joblib.dump(MetaCDconf_clf, cl_2)
        joblib.dump(MetaRPconf_clf, cl_3)
        joblib.dump(MetaRGconf_clf, cl_4)
        joblib.dump(MetaFLint_clf, cl_5)
        joblib.dump(MetaCDint_clf, cl_6)
        joblib.dump(MetaRPint_clf, cl_7)
        joblib.dump(MetaRGint_clf, cl_8)
        joblib.dump(MetaFLavail_clf, cl_9)
        joblib.dump(MetaCDavail_clf, cl_10)
        joblib.dump(MetaRPavail_clf, cl_11)
        joblib.dump(MetaRGavail_clf, cl_12)

        #save training features
        joblib.dump(constants.nMetaBaseNonText_feature, tf1)
        joblib.dump(constants.nMetaMetaNonText_feature, tf2)

        # save encoders
        joblib.dump(MetaFLconf_enc, en_1)
        joblib.dump(MetaCDconf_enc, en_2)
        joblib.dump(MetaRPconf_enc, en_3)
        joblib.dump(MetaRGconf_enc, en_4)
        joblib.dump(MetaFLint_enc, en_5)
        joblib.dump(MetaCDint_enc, en_6)
        joblib.dump(MetaRPint_enc, en_7)
        joblib.dump(MetaRGint_enc, en_8)
        joblib.dump(MetaFLavail_enc, en_9)
        joblib.dump(MetaCDavail_enc, en_10)
        joblib.dump(MetaRPavail_enc, en_11)
        joblib.dump(MetaRGavail_enc, en_12)

        #save metrics
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix

        #write to file
        text_file = open(outputMetrics, "w+")

        text_file.write('Best estimator found by GridSearch: \n')
        text_file.write('FL confidentiality estimator: %s \n\n' % MetaFLconf_clf.best_estimator_)
        text_file.write('CD confidentiality estimator: %s \n\n' % MetaCDconf_clf.best_estimator_)
        text_file.write('RP confidentiality estimator: %s \n\n' % MetaRPconf_clf.best_estimator_)
        text_file.write('RG confidentiality estimator: %s \n\n' % MetaRGconf_clf.best_estimator_)
        text_file.write('FL integrity estimator: %s \n\n' % MetaFLint_clf.best_estimator_)
        text_file.write('CD integrity estimator: %s \n\n' % MetaCDint_clf.best_estimator_)
        text_file.write('RP integrity estimator: %s \n\n' % MetaRPint_clf.best_estimator_)
        text_file.write('RG integrity estimator: %s \n\n' % MetaRGint_clf.best_estimator_)
        text_file.write('FL availability estimator: %s \n\n' % MetaFLavail_clf.best_estimator_)
        text_file.write('CD availability estimator: %s \n\n' % MetaCDavail_clf.best_estimator_)
        text_file.write('RP availability estimator: %s \n\n' % MetaRPavail_clf.best_estimator_)
        text_file.write('RG availability estimator: %s \n\n' % MetaRGavail_clf.best_estimator_)

        if constants.train_test_split != 1.0:
            text_file.write('Test Accuracy Predictions: \n')
            text_file.write('FL confidentiality prediction accuracy: %.7f \n' % classifier_acc[0])
            text_file.write('CD confidentiality prediction accuracy: %.7f \n' % classifier_acc[1])
            text_file.write('RP confidentiality prediction accuracy: %.7f \n' % classifier_acc[2])
            text_file.write('RG confidentiality prediction accuracy: %.7f \n' % classifier_acc[3])
            text_file.write('FL integrity prediction accuracy: %.7f \n' % classifier_acc[4])
            text_file.write('CD integrity prediction accuracy: %.7f \n' % classifier_acc[5])
            text_file.write('RP integrity prediction accuracy: %.7f \n' % classifier_acc[6])
            text_file.write('RG integrity prediction accuracy: %.7f \n' % classifier_acc[7])
            text_file.write('FL availability prediction accuracy: %.7f \n' % classifier_acc[8])
            text_file.write('CD availability prediction accuracy: %.7f \n' % classifier_acc[9])
            text_file.write('RP availability prediction accuracy: %.7f \n' % classifier_acc[10])
            text_file.write('RG availability prediction accuracy: %.7f \n' % classifier_acc[11])
            text_file.write('Overall accuracy: %.7f \n\n' % np.mean(classifier_acc))

        text_file.write('Train Accuracy Predictions: \n')
        text_file.write('FL confidentiality prediction accuracy: %.7f \n' % classifier_tacc[0])
        text_file.write('CD confidentiality prediction accuracy: %.7f \n' % classifier_tacc[1])
        text_file.write('RP confidentiality prediction accuracy: %.7f \n' % classifier_tacc[2])
        text_file.write('RG confidentiality prediction accuracy: %.7f \n' % classifier_tacc[3])
        text_file.write('FL integrity prediction accuracy: %.7f \n' % classifier_tacc[4])
        text_file.write('CD integrity prediction accuracy: %.7f \n' % classifier_tacc[5])
        text_file.write('RP integrity prediction accuracy: %.7f \n' % classifier_tacc[6])
        text_file.write('RG integrity prediction accuracy: %.7f \n' % classifier_tacc[7])
        text_file.write('FL availability prediction accuracy: %.7f \n' % classifier_tacc[8])
        text_file.write('CD availability prediction accuracy: %.7f \n' % classifier_tacc[9])
        text_file.write('RP availability prediction accuracy: %.7f \n' % classifier_tacc[10])
        text_file.write('RG availability prediction accuracy: %.7f \n' % classifier_tacc[11])
        text_file.write('Overall accuracy: %.7f \n\n\n' % np.mean(classifier_tacc))

        if constants.train_test_split != 1.0:
            text_file.write('FL confidentiality classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetatransLabelsTest[0],MetaFLconf_pred,
                                                            target_names=['L','M','H','VH']))
            text_file.write('CD confidentiality classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetatransLabelsTest[1], MetaCDconf_pred,
                                                            target_names=['L', 'M', 'H', 'VH']))
            text_file.write('RP confidentiality classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetatransLabelsTest[2], MetaRPconf_pred,
                                                            target_names=['L', 'M', 'H', 'VH']))
            text_file.write('RG confidentiality classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetatransLabelsTest[3], MetaRGconf_pred,
                                                              target_names=['L', 'M', 'H', 'VH']))
            text_file.write('FL integrity classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetatransLabelsTest[4], MetaFLint_pred,
                                                              target_names=['L', 'M', 'H', 'VH']))
            text_file.write('CD integrity classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetatransLabelsTest[5], MetaCDint_pred,
                                                              target_names=['L', 'M', 'H', 'VH']))
            text_file.write('RP integrity classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetatransLabelsTest[6], MetaRPint_pred,
                                                              target_names=['L', 'M', 'H', 'VH']))
            text_file.write('RG integrity classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetatransLabelsTest[7], MetaRGint_pred,
                                                              target_names=['L', 'M', 'H', 'VH']))
            text_file.write('FL availability classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetatransLabelsTest[8], MetaFLavail_pred,
                                                              target_names=['L', 'M', 'H', 'VH']))
            text_file.write('CD availability classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetatransLabelsTest[9], MetaCDavail_pred,
                                                              target_names=['L', 'M', 'H', 'VH']))
            text_file.write('RP availability classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetatransLabelsTest[10], MetaRPavail_pred,
                                                              target_names=['L', 'M', 'H', 'VH']))
            text_file.write('RG availability classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetatransLabelsTest[11], MetaRGavail_pred,
                                                              target_names=['L', 'M', 'H', 'VH']))

        text_file.close()
        print 'saving L2 training classifier data complete...'