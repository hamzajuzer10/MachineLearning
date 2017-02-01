#!/usr/bin/python

#Train machine learning algorithms on features and labels

import numpy as np
import constants
import sys
import os
import datetime
from time import time
import FeatureTrainingPreprocessing
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

# Run training
t0 = time()

if constants.meta:
    print 'training L1 classifier...'
else:
    print 'training classifier...'

label_clf = {}

for label_t,classifier_t in constants.classifier.items():

    if classifier_t == 'GaussianNaiveBayes':
        print 'training Gaussian Naive Bayes classifier on %s...' %label_t
        label_clf[label_t] = GridSearchCV(GaussianNB(),constants.param_grid[label_t])

    elif classifier_t == 'DecisionTree':
        print 'training Decision Tree classifier on %s...' %label_t
        label_clf[label_t] = GridSearchCV(DecisionTreeClassifier(),constants.param_grid[label_t])

    elif classifier_t == 'SVM':
        print 'training SVM classifier on %s...' %label_t
        label_clf[label_t] = GridSearchCV(SVC(),constants.param_grid[label_t])

    elif classifier_t == 'KNN':
        print 'training KNN classifier on %s...' %label_t
        label_clf[label_t] = GridSearchCV(KNeighborsClassifier(),constants.param_grid[label_t])

    elif classifier_t == 'RandomForest':
        print 'training RandomForest classifier on %s...' %label_t
        label_clf[label_t] = GridSearchCV(RandomForestClassifier(),constants.param_grid[label_t])

    elif classifier_t == 'AdaBoostDT':
        print 'training AdaBoost Decision Tree classifier on %s...' %label_t
        label_clf[label_t] = GridSearchCV(AdaBoostClassifier(base_estimator = DecisionTreeClassifier()),constants.param_grid[label_t])

    else:
        print 'no classifier selected...'
        print 'program is exiting...'
        sys.exit(0)

#fit classifier
FLconf_clf = label_clf['FLconfidentiality']
CDconf_clf = label_clf['CDconfidentiality']
RPconf_clf = label_clf['RPconfidentiality']
RGconf_clf = label_clf['RGconfidentiality']
FLint_clf = label_clf['FLintegrity']
CDint_clf = label_clf['CDintegrity']
RPint_clf = label_clf['RPintegrity']
RGint_clf = label_clf['RGintegrity']
FLavail_clf = label_clf['FLavailability']
CDavail_clf = label_clf['CDavailability']
RPavail_clf = label_clf['RPavailability']
RGavail_clf = label_clf['RGavailability']


FLconf_clf.fit(FeatureTrainingPreprocessing.FLtransConfFeaturesNonTextTrain,FeatureTrainingPreprocessing.transLabelsTrain[0])
print '1'
CDconf_clf.fit(FeatureTrainingPreprocessing.CDtransConfFeaturesNonTextTrain,FeatureTrainingPreprocessing.transLabelsTrain[1])
print '2'
RPconf_clf.fit(FeatureTrainingPreprocessing.RPtransConfFeaturesNonTextTrain,FeatureTrainingPreprocessing.transLabelsTrain[2])
print '3'
RGconf_clf.fit(FeatureTrainingPreprocessing.RGtransConfFeaturesNonTextTrain,FeatureTrainingPreprocessing.transLabelsTrain[3])
print '4'
FLint_clf.fit(FeatureTrainingPreprocessing.FLtransIntFeaturesNonTextTrain,FeatureTrainingPreprocessing.transLabelsTrain[4])
print '5'
CDint_clf.fit(FeatureTrainingPreprocessing.CDtransIntFeaturesNonTextTrain,FeatureTrainingPreprocessing.transLabelsTrain[5])
print '6'
RPint_clf.fit(FeatureTrainingPreprocessing.RPtransIntFeaturesNonTextTrain,FeatureTrainingPreprocessing.transLabelsTrain[6])
print '7'
RGint_clf.fit(FeatureTrainingPreprocessing.RGtransIntFeaturesNonTextTrain,FeatureTrainingPreprocessing.transLabelsTrain[7])
print '8'
FLavail_clf.fit(FeatureTrainingPreprocessing.FLtransAvailFeaturesNonTextTrain,FeatureTrainingPreprocessing.transLabelsTrain[8])
print '9'
CDavail_clf.fit(FeatureTrainingPreprocessing.CDtransAvailFeaturesNonTextTrain,FeatureTrainingPreprocessing.transLabelsTrain[9])
print '10'
RPavail_clf.fit(FeatureTrainingPreprocessing.RPtransAvailFeaturesNonTextTrain,FeatureTrainingPreprocessing.transLabelsTrain[10])
print '11'
RGavail_clf.fit(FeatureTrainingPreprocessing.RGtransAvailFeaturesNonTextTrain,FeatureTrainingPreprocessing.transLabelsTrain[11])


nb_training_time = round(time()-t0,3)

if constants.meta:
    print 'L1 training complete: ',nb_training_time, 's'
    print 'computing test and train prediction results for L1 classifier...'
else:
    print 'training complete: ', nb_training_time, 's'
    print 'computing test and train prediction results for classifier...'

if constants.train_test_split != 1.0:

    if constants.meta:
        print 'running test prediction on L1 classifier...'
    else:
        print 'running test prediction on classifier...'

    #compute test results
    FLconf_pred = FLconf_clf.predict(FeatureTrainingPreprocessing.FLtransConfFeaturesNonTextTest)
    CDconf_pred = CDconf_clf.predict(FeatureTrainingPreprocessing.CDtransConfFeaturesNonTextTest)
    RPconf_pred = RPconf_clf.predict(FeatureTrainingPreprocessing.RPtransConfFeaturesNonTextTest)
    RGconf_pred = RGconf_clf.predict(FeatureTrainingPreprocessing.RGtransConfFeaturesNonTextTest)
    FLint_pred = FLint_clf.predict(FeatureTrainingPreprocessing.FLtransIntFeaturesNonTextTest)
    CDint_pred = CDint_clf.predict(FeatureTrainingPreprocessing.CDtransIntFeaturesNonTextTest)
    RPint_pred = RPint_clf.predict(FeatureTrainingPreprocessing.RPtransIntFeaturesNonTextTest)
    RGint_pred = RGint_clf.predict(FeatureTrainingPreprocessing.RGtransIntFeaturesNonTextTest)
    FLavail_pred = FLavail_clf.predict(FeatureTrainingPreprocessing.FLtransAvailFeaturesNonTextTest)
    CDavail_pred = CDavail_clf.predict(FeatureTrainingPreprocessing.CDtransAvailFeaturesNonTextTest)
    RPavail_pred = RPavail_clf.predict(FeatureTrainingPreprocessing.RPtransAvailFeaturesNonTextTest)
    RGavail_pred = RGavail_clf.predict(FeatureTrainingPreprocessing.RGtransAvailFeaturesNonTextTest)


    print 'test prediction complete...'

    #compute test accuracy
    classifier_acc=[]
    classifier_acc.append(accuracy_score(FLconf_pred, FeatureTrainingPreprocessing.transLabelsTest[0]))
    classifier_acc.append(accuracy_score(CDconf_pred, FeatureTrainingPreprocessing.transLabelsTest[1]))
    classifier_acc.append(accuracy_score(RPconf_pred, FeatureTrainingPreprocessing.transLabelsTest[2]))
    classifier_acc.append(accuracy_score(RGconf_pred, FeatureTrainingPreprocessing.transLabelsTest[3]))
    classifier_acc.append(accuracy_score(FLint_pred, FeatureTrainingPreprocessing.transLabelsTest[4]))
    classifier_acc.append(accuracy_score(CDint_pred, FeatureTrainingPreprocessing.transLabelsTest[5]))
    classifier_acc.append(accuracy_score(RPint_pred, FeatureTrainingPreprocessing.transLabelsTest[6]))
    classifier_acc.append(accuracy_score(RGint_pred, FeatureTrainingPreprocessing.transLabelsTest[7]))
    classifier_acc.append(accuracy_score(FLavail_pred, FeatureTrainingPreprocessing.transLabelsTest[8]))
    classifier_acc.append(accuracy_score(CDavail_pred, FeatureTrainingPreprocessing.transLabelsTest[9]))
    classifier_acc.append(accuracy_score(RPavail_pred, FeatureTrainingPreprocessing.transLabelsTest[10]))
    classifier_acc.append(accuracy_score(RGavail_pred, FeatureTrainingPreprocessing.transLabelsTest[11]))

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

if constants.meta:
    print 'running training prediction on L1 classifier...'
else:
    print 'running training prediction on classifier...'

# computing train results for bias/variance score
FLconf_trpred = FLconf_clf.predict(FeatureTrainingPreprocessing.FLtransConfFeaturesNonTextTrain)
CDconf_trpred = CDconf_clf.predict(FeatureTrainingPreprocessing.CDtransConfFeaturesNonTextTrain)
RPconf_trpred = RPconf_clf.predict(FeatureTrainingPreprocessing.RPtransConfFeaturesNonTextTrain)
RGconf_trpred = RGconf_clf.predict(FeatureTrainingPreprocessing.RGtransConfFeaturesNonTextTrain)
FLint_trpred = FLint_clf.predict(FeatureTrainingPreprocessing.FLtransIntFeaturesNonTextTrain)
CDint_trpred = CDint_clf.predict(FeatureTrainingPreprocessing.CDtransIntFeaturesNonTextTrain)
RPint_trpred = RPint_clf.predict(FeatureTrainingPreprocessing.RPtransIntFeaturesNonTextTrain)
RGint_trpred = RGint_clf.predict(FeatureTrainingPreprocessing.RGtransIntFeaturesNonTextTrain)
FLavail_trpred = FLavail_clf.predict(FeatureTrainingPreprocessing.FLtransAvailFeaturesNonTextTrain)
CDavail_trpred = CDavail_clf.predict(FeatureTrainingPreprocessing.CDtransAvailFeaturesNonTextTrain)
RPavail_trpred = RPavail_clf.predict(FeatureTrainingPreprocessing.RPtransAvailFeaturesNonTextTrain)
RGavail_trpred = RGavail_clf.predict(FeatureTrainingPreprocessing.RGtransAvailFeaturesNonTextTrain)

print 'training prediction complete...'

# compute train accuracy
classifier_tacc = []
classifier_tacc.append(accuracy_score(FLconf_trpred, FeatureTrainingPreprocessing.transLabelsTrain[0]))
classifier_tacc.append(accuracy_score(CDconf_trpred, FeatureTrainingPreprocessing.transLabelsTrain[1]))
classifier_tacc.append(accuracy_score(RPconf_trpred, FeatureTrainingPreprocessing.transLabelsTrain[2]))
classifier_tacc.append(accuracy_score(RGconf_trpred, FeatureTrainingPreprocessing.transLabelsTrain[3]))
classifier_tacc.append(accuracy_score(FLint_trpred, FeatureTrainingPreprocessing.transLabelsTrain[4]))
classifier_tacc.append(accuracy_score(CDint_trpred, FeatureTrainingPreprocessing.transLabelsTrain[5]))
classifier_tacc.append(accuracy_score(RPint_trpred, FeatureTrainingPreprocessing.transLabelsTrain[6]))
classifier_tacc.append(accuracy_score(RGint_trpred, FeatureTrainingPreprocessing.transLabelsTrain[7]))
classifier_tacc.append(accuracy_score(FLavail_trpred, FeatureTrainingPreprocessing.transLabelsTrain[8]))
classifier_tacc.append(accuracy_score(CDavail_trpred, FeatureTrainingPreprocessing.transLabelsTrain[9]))
classifier_tacc.append(accuracy_score(RPavail_trpred, FeatureTrainingPreprocessing.transLabelsTrain[10]))
classifier_tacc.append(accuracy_score(RGavail_trpred, FeatureTrainingPreprocessing.transLabelsTrain[11]))

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


if constants.meta:

    print 'training L2 classifier...'

    #create predicted array of test results
    mconf_pred = []
    mconf_pred.append(FLconf_pred)
    mconf_pred.append(CDconf_pred)
    mconf_pred.append(RPconf_pred)
    mconf_pred.append(RGconf_pred)
    mconf_pred.append(FLint_pred)
    mconf_pred.append(CDint_pred)
    mconf_pred.append(RPint_pred)
    mconf_pred.append(RGint_pred)
    mconf_pred.append(FLavail_pred)
    mconf_pred.append(CDavail_pred)
    mconf_pred.append(RPavail_pred)
    mconf_pred.append(RGavail_pred)


    # create predicted array of training results
    mconf_trpred = []
    mconf_trpred.append(FLconf_pred)
    mconf_trpred.append(CDconf_pred)
    mconf_trpred.append(RPconf_pred)
    mconf_trpred.append(RGconf_pred)
    mconf_trpred.append(FLint_pred)
    mconf_trpred.append(CDint_pred)
    mconf_trpred.append(RPint_pred)
    mconf_trpred.append(RGint_pred)
    mconf_trpred.append(FLavail_pred)
    mconf_trpred.append(CDavail_pred)
    mconf_trpred.append(RPavail_pred)
    mconf_trpred.append(RGavail_pred)


if constants.save:
    print 'saving training classifier data...'

    from sklearn.externals import joblib
    directory = 'Outputs/'
    if constants.train_test_split != 1.0:
        directory += str(int(np.mean(classifier_acc)*100))
        directory += '%_acc_'
    directory += 'TrainingSet_'
    directory += str(len(FeatureTrainingPreprocessing.FeatureTrainingExtraction.training_dict_list))
    directory += '_'
    directory += str(datetime.date.today())
    directory += '_'
    directory += str('%02dh' %datetime.datetime.now().time().hour)
    directory += str('%02dm' % datetime.datetime.now().time().minute)



    #make directory if one doesnt exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    #filename (clfs)
    cl_1 = directory + str('/FLconf.pkl')
    cl_2 = directory + str('/CDconf.pkl')
    cl_3 = directory + str('/RPconf.pkl')
    cl_4 = directory + str('/RGconf.pkl')
    cl_5 = directory + str('/FLint.pkl')
    cl_6 = directory + str('/CDint.pkl')
    cl_7 = directory + str('/RPint.pkl')
    cl_8 = directory + str('/RGint.pkl')
    cl_9 = directory + str('/FLavail.pkl')
    cl_10= directory + str('/CDavail.pkl')
    cl_11= directory + str('/RPavail.pkl')
    cl_12= directory + str('/RGavail.pkl')

    #filename (output metrics)
    outputMetrics = directory +str('/metrics.txt')

    #filename (features)
    tf = directory + str('/nNonTextFeature.pkl')

    #filename (encoders)
    en_1 = directory + str('/FLconfEncoder.pkl')
    en_2 = directory + str('/CDconfEncoder.pkl')
    en_3 = directory + str('/RPconfEncoder.pkl')
    en_4 = directory + str('/RGconfEncoder.pkl')
    en_5 = directory + str('/FLintEncoder.pkl')
    en_6 = directory + str('/CDintEncoder.pkl')
    en_7 = directory + str('/RPintEncoder.pkl')
    en_8 = directory + str('/RGintEncoder.pkl')
    en_9 = directory + str('/FLavailEncoder.pkl')
    en_10 = directory + str('/CDavailEncoder.pkl')
    en_11 = directory + str('/RPavailEncoder.pkl')
    en_12 = directory + str('/RGavailEncoder.pkl')

    #save clfs
    joblib.dump(FLconf_clf, cl_1)
    joblib.dump(CDconf_clf, cl_2)
    joblib.dump(RPconf_clf, cl_3)
    joblib.dump(RGconf_clf, cl_4)
    joblib.dump(FLint_clf, cl_5)
    joblib.dump(CDint_clf, cl_6)
    joblib.dump(RPint_clf, cl_7)
    joblib.dump(RGint_clf, cl_8)
    joblib.dump(FLavail_clf, cl_9)
    joblib.dump(CDavail_clf, cl_10)
    joblib.dump(RPavail_clf, cl_11)
    joblib.dump(RGavail_clf, cl_12)

    #save training features
    joblib.dump(constants.nNonText_feature, tf)

    # save encoders
    joblib.dump(FeatureTrainingPreprocessing.FLconf_enc, en_1)
    joblib.dump(FeatureTrainingPreprocessing.CDconf_enc, en_2)
    joblib.dump(FeatureTrainingPreprocessing.RPconf_enc, en_3)
    joblib.dump(FeatureTrainingPreprocessing.RGconf_enc, en_4)
    joblib.dump(FeatureTrainingPreprocessing.FLint_enc, en_5)
    joblib.dump(FeatureTrainingPreprocessing.CDint_enc, en_6)
    joblib.dump(FeatureTrainingPreprocessing.RPint_enc, en_7)
    joblib.dump(FeatureTrainingPreprocessing.RGint_enc, en_8)
    joblib.dump(FeatureTrainingPreprocessing.FLavail_enc, en_9)
    joblib.dump(FeatureTrainingPreprocessing.CDavail_enc, en_10)
    joblib.dump(FeatureTrainingPreprocessing.RPavail_enc, en_11)
    joblib.dump(FeatureTrainingPreprocessing.RGavail_enc, en_12)

    #save metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

    #write to file
    text_file = open(outputMetrics, "w+")

    text_file.write('Best estimator found by GridSearch: \n')
    text_file.write('FL confidentiality estimator: %s \n\n' % FLconf_clf.best_estimator_)
    text_file.write('CD confidentiality estimator: %s \n\n' % CDconf_clf.best_estimator_)
    text_file.write('RP confidentiality estimator: %s \n\n' % RPconf_clf.best_estimator_)
    text_file.write('RG confidentiality estimator: %s \n\n' % RGconf_clf.best_estimator_)
    text_file.write('FL integrity estimator: %s \n\n' % FLint_clf.best_estimator_)
    text_file.write('CD integrity estimator: %s \n\n' % CDint_clf.best_estimator_)
    text_file.write('RP integrity estimator: %s \n\n' % RPint_clf.best_estimator_)
    text_file.write('RG integrity estimator: %s \n\n' % RGint_clf.best_estimator_)
    text_file.write('FL availability estimator: %s \n\n' % FLavail_clf.best_estimator_)
    text_file.write('CD availability estimator: %s \n\n' % CDavail_clf.best_estimator_)
    text_file.write('RP availability estimator: %s \n\n' % RPavail_clf.best_estimator_)
    text_file.write('RG availability estimator: %s \n\n' % RGavail_clf.best_estimator_)

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
        text_file.write('%s \n\n' % classification_report(FeatureTrainingPreprocessing.transLabelsTest[0],FLconf_pred,
                                                        target_names=['L','M','H','VH']))
        text_file.write('CD confidentiality classification report:\n')
        text_file.write('%s \n\n' % classification_report(FeatureTrainingPreprocessing.transLabelsTest[1], CDconf_pred,
                                                        target_names=['L', 'M', 'H', 'VH']))
        text_file.write('RP confidentiality classification report:\n')
        text_file.write('%s \n\n' % classification_report(FeatureTrainingPreprocessing.transLabelsTest[2], RPconf_pred,
                                                        target_names=['L', 'M', 'H', 'VH']))
        text_file.write('RG confidentiality classification report:\n')
        text_file.write('%s \n\n' % classification_report(FeatureTrainingPreprocessing.transLabelsTest[3], RGconf_pred,
                                                          target_names=['L', 'M', 'H', 'VH']))
        text_file.write('FL integrity classification report:\n')
        text_file.write('%s \n\n' % classification_report(FeatureTrainingPreprocessing.transLabelsTest[4], FLint_pred,
                                                          target_names=['L', 'M', 'H', 'VH']))
        text_file.write('CD integrity classification report:\n')
        text_file.write('%s \n\n' % classification_report(FeatureTrainingPreprocessing.transLabelsTest[5], CDint_pred,
                                                          target_names=['L', 'M', 'H', 'VH']))
        text_file.write('RP integrity classification report:\n')
        text_file.write('%s \n\n' % classification_report(FeatureTrainingPreprocessing.transLabelsTest[6], RPint_pred,
                                                          target_names=['L', 'M', 'H', 'VH']))
        text_file.write('RG integrity classification report:\n')
        text_file.write('%s \n\n' % classification_report(FeatureTrainingPreprocessing.transLabelsTest[7], RGint_pred,
                                                          target_names=['L', 'M', 'H', 'VH']))
        text_file.write('FL availability classification report:\n')
        text_file.write('%s \n\n' % classification_report(FeatureTrainingPreprocessing.transLabelsTest[8], FLavail_pred,
                                                          target_names=['L', 'M', 'H', 'VH']))
        text_file.write('CD availability classification report:\n')
        text_file.write('%s \n\n' % classification_report(FeatureTrainingPreprocessing.transLabelsTest[9], CDavail_pred,
                                                          target_names=['L', 'M', 'H', 'VH']))
        text_file.write('RP availability classification report:\n')
        text_file.write('%s \n\n' % classification_report(FeatureTrainingPreprocessing.transLabelsTest[10], RPavail_pred,
                                                          target_names=['L', 'M', 'H', 'VH']))
        text_file.write('RG availability classification report:\n')
        text_file.write('%s \n\n' % classification_report(FeatureTrainingPreprocessing.transLabelsTest[11], RGavail_pred,
                                                          target_names=['L', 'M', 'H', 'VH']))

    text_file.close()
    print 'saving training classifier data complete...'


