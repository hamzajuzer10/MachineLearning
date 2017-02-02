#!/usr/bin/python

#Train machine learning algorithms on features and labels

import numpy as np
import constants
import imp
import sys
import os
from time import time
import import FeatureTrainingExtraction
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from PreprocessingFunctions import label_transform, metaNonTextFeature_combine, nonTextFeature_transform, nonTextFeature_nvalues, checkLabelsNFeatures
from sklearn.preprocessing import OneHotEncoder


def MetaTrainAndTest(mconf_pred, mconf_trpred):

    # Run training
    t0 = time()

    print 'training L2 classifier...'

    label_clf = {}

    for label_t,classifier_t in constants.meta_classifier.items():

        if classifier_t == 'GaussianNaiveBayes':
            print 'training Gaussian Naive Bayes classifier on %s...' %label_t
            label_clf[label_t] = GridSearchCV(GaussianNB(),constants.meta_param_grid[label_t])

        elif classifier_t == 'DecisionTree':
            print 'training Decision Tree classifier on %s...' %label_t
            label_clf[label_t] = GridSearchCV(DecisionTreeClassifier(),constants.meta_param_grid[label_t])

        elif classifier_t == 'SVM':
            print 'training SVM classifier on %s...' %label_t
            label_clf[label_t] = GridSearchCV(SVC(),constants.meta_param_grid[label_t])

        elif classifier_t == 'KNN':
            print 'training KNN classifier on %s...' %label_t
            label_clf[label_t] = GridSearchCV(KNeighborsClassifier(),constants.meta_param_grid[label_t])

        elif classifier_t == 'RandomForest':
            print 'training RandomForest classifier on %s...' %label_t
            label_clf[label_t] = GridSearchCV(RandomForestClassifier(),constants.meta_param_grid[label_t])

        elif classifier_t == 'AdaBoostDT':
            print 'training AdaBoost Decision Tree classifier on %s...' %label_t
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

    if imp.lock_held():
        imp.release_lock()

        print 'training %s L2 classifier on Confidentiality FinLoss...' % constants.meta_classifier['FLconfidentiality']
        MetaFLconf_clf.fit(MetaFeatureTrainingPreprocessing.MetaFLtransConfFeaturesNonTextTrain,MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[0])

        print 'training %s L2 classifier on Confidentiality CusD...' % constants.meta_classifier['CDconfidentiality']
        MetaCDconf_clf.fit(MetaFeatureTrainingPreprocessing.MetaCDtransConfFeaturesNonTextTrain,MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[1])

        print 'training %s L2 classifier on Confidentiality RepLoss...' % constants.meta_classifier['RPconfidentiality']
        MetaRPconf_clf.fit(MetaFeatureTrainingPreprocessing.MetaRPtransConfFeaturesNonTextTrain,MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[2])

        print 'training %s L2 classifier on Confidentiality RegLoss...' % constants.meta_classifier['RGconfidentiality']
        MetaRGconf_clf.fit(MetaFeatureTrainingPreprocessing.MetaRGtransConfFeaturesNonTextTrain,MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[3])

        print 'training %s L2 classifier on Integrity FinLoss...' % constants.meta_classifier['FLintegrity']
        MetaFLint_clf.fit(MetaFeatureTrainingPreprocessing.MetaFLtransIntFeaturesNonTextTrain,MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[4])

        print 'training %s L2 classifier on Integrity CusD...' % constants.meta_classifier['CDintegrity']
        MetaCDint_clf.fit(MetaFeatureTrainingPreprocessing.MetaCDtransIntFeaturesNonTextTrain,MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[5])

        print 'training %s L2 classifier on Integrity RepLoss...' % constants.meta_classifier['RPintegrity']
        MetaRPint_clf.fit(MetaFeatureTrainingPreprocessing.MetaRPtransIntFeaturesNonTextTrain,MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[6])

        print 'training %s L2 classifier on Integrity RegLoss...' % constants.meta_classifier['RGintegrity']
        MetaRGint_clf.fit(MetaFeatureTrainingPreprocessing.MetaRGtransIntFeaturesNonTextTrain,MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[7])

        print 'training %s L2 classifier on Availability FinLoss...' % constants.meta_classifier['FLavailability']
        MetaFLavail_clf.fit(MetaFeatureTrainingPreprocessing.MetaFLtransAvailFeaturesNonTextTrain,MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[8])

        print 'training %s L2 classifier on Availability CusD...' % constants.meta_classifier['CDavailability']
        MetaCDavail_clf.fit(MetaFeatureTrainingPreprocessing.MetaCDtransAvailFeaturesNonTextTrain,MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[9])

        print 'training %s L2 classifier on Availability RepLoss...' % constants.meta_classifier['RPavailability']
        MetaRPavail_clf.fit(MetaFeatureTrainingPreprocessing.MetaRPtransAvailFeaturesNonTextTrain,MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[10])

        print 'training %s L2 classifier on Availability RegLoss...' % constants.meta_classifier['RGavailability']
        MetaRGavail_clf.fit(MetaFeatureTrainingPreprocessing.MetaRGtransAvailFeaturesNonTextTrain,MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[11])

        imp.acquire_lock()


    nb_training_time = round(time()-t0,3)

    print 'L2 training complete: ',nb_training_time, 's'
    print 'computing test and train prediction results for L2 classifier...'

    if constants.train_test_split != 1.0:

        print 'running test prediction on L2 classifier...'

        #compute test results
        if imp.lock_held():
            imp.release_lock()
            MetaFLconf_pred = MetaFLconf_clf.predict(MetaFeatureTrainingPreprocessing.MetaFLtransConfFeaturesNonTextTest)
            MetaCDconf_pred = MetaCDconf_clf.predict(MetaFeatureTrainingPreprocessing.MetaCDtransConfFeaturesNonTextTest)
            MetaRPconf_pred = MetaRPconf_clf.predict(MetaFeatureTrainingPreprocessing.MetaRPtransConfFeaturesNonTextTest)
            MetaRGconf_pred = MetaRGconf_clf.predict(MetaFeatureTrainingPreprocessing.MetaRGtransConfFeaturesNonTextTest)
            MetaFLint_pred = MetaFLint_clf.predict(MetaFeatureTrainingPreprocessing.MetaFLtransIntFeaturesNonTextTest)
            MetaCDint_pred = MetaCDint_clf.predict(MetaFeatureTrainingPreprocessing.MetaCDtransIntFeaturesNonTextTest)
            MetaRPint_pred = MetaRPint_clf.predict(MetaFeatureTrainingPreprocessing.MetaRPtransIntFeaturesNonTextTest)
            MetaRGint_pred = MetaRGint_clf.predict(MetaFeatureTrainingPreprocessing.MetaRGtransIntFeaturesNonTextTest)
            MetaFLavail_pred = MetaFLavail_clf.predict(MetaFeatureTrainingPreprocessing.MetaFLtransAvailFeaturesNonTextTest)
            MetaCDavail_pred = MetaCDavail_clf.predict(MetaFeatureTrainingPreprocessing.MetaCDtransAvailFeaturesNonTextTest)
            MetaRPavail_pred = MetaRPavail_clf.predict(MetaFeatureTrainingPreprocessing.MetaRPtransAvailFeaturesNonTextTest)
            MetaRGavail_pred = MetaRGavail_clf.predict(MetaFeatureTrainingPreprocessing.MetaRGtransAvailFeaturesNonTextTest)
            imp.acquire_lock()


        print 'test prediction complete...'

        #compute test accuracy
        classifier_acc=[]
        classifier_acc.append(accuracy_score(MetaFLconf_pred, MetaFeatureTrainingPreprocessing.MetatransLabelsTest[0]))
        classifier_acc.append(accuracy_score(MetaCDconf_pred, MetaFeatureTrainingPreprocessing.MetatransLabelsTest[1]))
        classifier_acc.append(accuracy_score(MetaRPconf_pred, MetaFeatureTrainingPreprocessing.MetatransLabelsTest[2]))
        classifier_acc.append(accuracy_score(MetaRGconf_pred, MetaFeatureTrainingPreprocessing.MetatransLabelsTest[3]))
        classifier_acc.append(accuracy_score(MetaFLint_pred, MetaFeatureTrainingPreprocessing.MetatransLabelsTest[4]))
        classifier_acc.append(accuracy_score(MetaCDint_pred, MetaFeatureTrainingPreprocessing.MetatransLabelsTest[5]))
        classifier_acc.append(accuracy_score(MetaRPint_pred, MetaFeatureTrainingPreprocessing.MetatransLabelsTest[6]))
        classifier_acc.append(accuracy_score(MetaRGint_pred, MetaFeatureTrainingPreprocessing.MetatransLabelsTest[7]))
        classifier_acc.append(accuracy_score(MetaFLavail_pred, MetaFeatureTrainingPreprocessing.MetatransLabelsTest[8]))
        classifier_acc.append(accuracy_score(MetaCDavail_pred, MetaFeatureTrainingPreprocessing.MetatransLabelsTest[9]))
        classifier_acc.append(accuracy_score(MetaRPavail_pred, MetaFeatureTrainingPreprocessing.MetatransLabelsTest[10]))
        classifier_acc.append(accuracy_score(MetaRGavail_pred, MetaFeatureTrainingPreprocessing.MetatransLabelsTest[11]))

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
    if imp.lock_held():
        imp.release_lock()
        MetaFLconf_trpred = MetaFLconf_clf.predict(MetaFeatureTrainingPreprocessing.MetaFLtransConfFeaturesNonTextTrain)
        MetaCDconf_trpred = MetaCDconf_clf.predict(MetaFeatureTrainingPreprocessing.MetaCDtransConfFeaturesNonTextTrain)
        MetaRPconf_trpred = MetaRPconf_clf.predict(MetaFeatureTrainingPreprocessing.MetaRPtransConfFeaturesNonTextTrain)
        MetaRGconf_trpred = MetaRGconf_clf.predict(MetaFeatureTrainingPreprocessing.MetaRGtransConfFeaturesNonTextTrain)
        MetaFLint_trpred = MetaFLint_clf.predict(MetaFeatureTrainingPreprocessing.MetaFLtransIntFeaturesNonTextTrain)
        MetaCDint_trpred = MetaCDint_clf.predict(MetaFeatureTrainingPreprocessing.MetaCDtransIntFeaturesNonTextTrain)
        MetaRPint_trpred = MetaRPint_clf.predict(MetaFeatureTrainingPreprocessing.MetaRPtransIntFeaturesNonTextTrain)
        MetaRGint_trpred = MetaRGint_clf.predict(MetaFeatureTrainingPreprocessing.MetaRGtransIntFeaturesNonTextTrain)
        MetaFLavail_trpred = MetaFLavail_clf.predict(MetaFeatureTrainingPreprocessing.MetaFLtransAvailFeaturesNonTextTrain)
        MetaCDavail_trpred = MetaCDavail_clf.predict(MetaFeatureTrainingPreprocessing.MetaCDtransAvailFeaturesNonTextTrain)
        MetaRPavail_trpred = MetaRPavail_clf.predict(MetaFeatureTrainingPreprocessing.MetaRPtransAvailFeaturesNonTextTrain)
        MetaRGavail_trpred = MetaRGavail_clf.predict(MetaFeatureTrainingPreprocessing.MetaRGtransAvailFeaturesNonTextTrain)
        imp.acquire_lock()

    print 'training prediction complete...'

    # compute train accuracy
    classifier_tacc = []
    classifier_tacc.append(accuracy_score(MetaFLconf_trpred, MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[0]))
    classifier_tacc.append(accuracy_score(MetaCDconf_trpred, MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[1]))
    classifier_tacc.append(accuracy_score(MetaRPconf_trpred, MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[2]))
    classifier_tacc.append(accuracy_score(MetaRGconf_trpred, MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[3]))
    classifier_tacc.append(accuracy_score(MetaFLint_trpred, MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[4]))
    classifier_tacc.append(accuracy_score(MetaCDint_trpred, MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[5]))
    classifier_tacc.append(accuracy_score(MetaRPint_trpred, MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[6]))
    classifier_tacc.append(accuracy_score(MetaRGint_trpred, MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[7]))
    classifier_tacc.append(accuracy_score(MetaFLavail_trpred, MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[8]))
    classifier_tacc.append(accuracy_score(MetaCDavail_trpred, MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[9]))
    classifier_tacc.append(accuracy_score(MetaRPavail_trpred, MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[10]))
    classifier_tacc.append(accuracy_score(MetaRGavail_trpred, MetaFeatureTrainingPreprocessing.MetatransLabelsTrain[11]))

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
        directory = 'Outputs/'
        if constants.train_test_split != 1.0:
            directory += str(int(np.mean(classifier_acc) * 100))
            directory += '%_acc_'
        directory += 'MetaTrainingSet_'
        directory += str(len(MetaFeatureTrainingPreprocessing.FeatureTrainingExtraction.training_dict_list))
        directory += '_'
        directory += str(datetime.date.today())
        directory += '_'
        directory += str('%02dh' % datetime.datetime.now().time().hour)
        directory += str('%02dm' % datetime.datetime.now().time().minute)

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
        joblib.dump(MetaFeatureTrainingPreprocessing.MetaFLconf_enc, en_1)
        joblib.dump(MetaFeatureTrainingPreprocessing.MetaCDconf_enc, en_2)
        joblib.dump(MetaFeatureTrainingPreprocessing.MetaRPconf_enc, en_3)
        joblib.dump(MetaFeatureTrainingPreprocessing.MetaRGconf_enc, en_4)
        joblib.dump(MetaFeatureTrainingPreprocessing.MetaFLint_enc, en_5)
        joblib.dump(MetaFeatureTrainingPreprocessing.MetaCDint_enc, en_6)
        joblib.dump(MetaFeatureTrainingPreprocessing.MetaRPint_enc, en_7)
        joblib.dump(MetaFeatureTrainingPreprocessing.MetaRGint_enc, en_8)
        joblib.dump(MetaFeatureTrainingPreprocessing.MetaFLavail_enc, en_9)
        joblib.dump(MetaFeatureTrainingPreprocessing.MetaCDavail_enc, en_10)
        joblib.dump(MetaFeatureTrainingPreprocessing.MetaRPavail_enc, en_11)
        joblib.dump(MetaFeatureTrainingPreprocessing.MetaRGavail_enc, en_12)

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
            text_file.write('%s \n\n' % classification_report(MetaFeatureTrainingPreprocessing.MetatransLabelsTest[0],MetaFLconf_pred,
                                                            target_names=['L','M','H','VH']))
            text_file.write('CD confidentiality classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetaFeatureTrainingPreprocessing.MetatransLabelsTest[1], MetaCDconf_pred,
                                                            target_names=['L', 'M', 'H', 'VH']))
            text_file.write('RP confidentiality classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetaFeatureTrainingPreprocessing.MetatransLabelsTest[2], MetaRPconf_pred,
                                                            target_names=['L', 'M', 'H', 'VH']))
            text_file.write('RG confidentiality classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetaFeatureTrainingPreprocessing.MetatransLabelsTest[3], MetaRGconf_pred,
                                                              target_names=['L', 'M', 'H', 'VH']))
            text_file.write('FL integrity classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetaFeatureTrainingPreprocessing.MetatransLabelsTest[4], MetaFLint_pred,
                                                              target_names=['L', 'M', 'H', 'VH']))
            text_file.write('CD integrity classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetaFeatureTrainingPreprocessing.MetatransLabelsTest[5], MetaCDint_pred,
                                                              target_names=['L', 'M', 'H', 'VH']))
            text_file.write('RP integrity classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetaFeatureTrainingPreprocessing.MetatransLabelsTest[6], MetaRPint_pred,
                                                              target_names=['L', 'M', 'H', 'VH']))
            text_file.write('RG integrity classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetaFeatureTrainingPreprocessing.MetatransLabelsTest[7], MetaRGint_pred,
                                                              target_names=['L', 'M', 'H', 'VH']))
            text_file.write('FL availability classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetaFeatureTrainingPreprocessing.MetatransLabelsTest[8], MetaFLavail_pred,
                                                              target_names=['L', 'M', 'H', 'VH']))
            text_file.write('CD availability classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetaFeatureTrainingPreprocessing.MetatransLabelsTest[9], MetaCDavail_pred,
                                                              target_names=['L', 'M', 'H', 'VH']))
            text_file.write('RP availability classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetaFeatureTrainingPreprocessing.MetatransLabelsTest[10], MetaRPavail_pred,
                                                              target_names=['L', 'M', 'H', 'VH']))
            text_file.write('RG availability classification report:\n')
            text_file.write('%s \n\n' % classification_report(MetaFeatureTrainingPreprocessing.MetatransLabelsTest[11], MetaRGavail_pred,
                                                              target_names=['L', 'M', 'H', 'VH']))

        text_file.close()
        print 'saving L2 training classifier data complete...'