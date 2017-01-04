#!/usr/bin/python

#Train machine learning algorithms on features and labels

import numpy as np
import constants
import sys
import os
import datetime
from time import time

def runTrainingAndTesting(classifier,save):

    import FeaturePreprocessing
    from sklearn.metrics import accuracy_score

    # Run training
    t0 = time()

    #Naive Bayes
    if classifier == 'GaussianNaiveBayes':

        print 'training Gaussian Naive Bayes classifier...'
        from sklearn.naive_bayes import GaussianNB

        #init classifier for each of the 12 categories
        FLconf_clf = GaussianNB()
        CDconf_clf = GaussianNB()
        RPconf_clf = GaussianNB()
        RGconf_clf = GaussianNB()
        FLint_clf = GaussianNB()
        CDint_clf = GaussianNB()
        RPint_clf = GaussianNB()
        RGint_clf = GaussianNB()
        FLavail_clf = GaussianNB()
        CDavail_clf = GaussianNB()
        RPavail_clf = GaussianNB()
        RGavail_clf = GaussianNB()

    #decision tree
    elif classifier == 'DecisionTree':

        print 'training Decision Tree classifier...'
        from sklearn.tree import DecisionTreeClassifier

        # init classifier for each of the 12 categories
        FLconf_clf = DecisionTreeClassifier(min_samples_split=constants.min_samples_split)
        CDconf_clf = DecisionTreeClassifier(min_samples_split=constants.min_samples_split)
        RPconf_clf = DecisionTreeClassifier(min_samples_split=constants.min_samples_split)
        RGconf_clf = DecisionTreeClassifier(min_samples_split=constants.min_samples_split)
        FLint_clf = DecisionTreeClassifier(min_samples_split=constants.min_samples_split)
        CDint_clf = DecisionTreeClassifier(min_samples_split=constants.min_samples_split)
        RPint_clf = DecisionTreeClassifier(min_samples_split=constants.min_samples_split)
        RGint_clf = DecisionTreeClassifier(min_samples_split=constants.min_samples_split)
        FLavail_clf = DecisionTreeClassifier(min_samples_split=constants.min_samples_split)
        CDavail_clf = DecisionTreeClassifier(min_samples_split=constants.min_samples_split)
        RPavail_clf = DecisionTreeClassifier(min_samples_split=constants.min_samples_split)
        RGavail_clf = DecisionTreeClassifier(min_samples_split=constants.min_samples_split)

    else:

        print 'no classifier selected...'
        print 'program is exiting...'
        sys.exit(0)

    #fit classifier
    FLconf_clf.fit(FeaturePreprocessing.FLtransConfFeaturesNonTextTrain,FeaturePreprocessing.transLabelsTrain[0])
    CDconf_clf.fit(FeaturePreprocessing.CDtransConfFeaturesNonTextTrain,FeaturePreprocessing.transLabelsTrain[1])
    RPconf_clf.fit(FeaturePreprocessing.RPtransConfFeaturesNonTextTrain,FeaturePreprocessing.transLabelsTrain[2])
    RGconf_clf.fit(FeaturePreprocessing.RGtransConfFeaturesNonTextTrain,FeaturePreprocessing.transLabelsTrain[3])
    FLint_clf.fit(FeaturePreprocessing.FLtransIntFeaturesNonTextTrain,FeaturePreprocessing.transLabelsTrain[4])
    CDint_clf.fit(FeaturePreprocessing.CDtransIntFeaturesNonTextTrain,FeaturePreprocessing.transLabelsTrain[5])
    RPint_clf.fit(FeaturePreprocessing.RPtransIntFeaturesNonTextTrain,FeaturePreprocessing.transLabelsTrain[6])
    RGint_clf.fit(FeaturePreprocessing.RGtransIntFeaturesNonTextTrain,FeaturePreprocessing.transLabelsTrain[7])
    FLavail_clf.fit(FeaturePreprocessing.FLtransAvailFeaturesNonTextTrain,FeaturePreprocessing.transLabelsTrain[8])
    CDavail_clf.fit(FeaturePreprocessing.CDtransAvailFeaturesNonTextTrain,FeaturePreprocessing.transLabelsTrain[9])
    RPavail_clf.fit(FeaturePreprocessing.RPtransAvailFeaturesNonTextTrain,FeaturePreprocessing.transLabelsTrain[10])
    RGavail_clf.fit(FeaturePreprocessing.RGtransAvailFeaturesNonTextTrain,FeaturePreprocessing.transLabelsTrain[11])


    nb_training_time = round(time()-t0,3)

    print 'training complete: ',nb_training_time, 's'
    print 'running test prediction on classifier...'

    #compute results
    FLconf_pred = FLconf_clf.predict(FeaturePreprocessing.FLtransConfFeaturesNonTextTest)
    CDconf_pred = CDconf_clf.predict(FeaturePreprocessing.CDtransConfFeaturesNonTextTest)
    RPconf_pred = RPconf_clf.predict(FeaturePreprocessing.RPtransConfFeaturesNonTextTest)
    RGconf_pred = RGconf_clf.predict(FeaturePreprocessing.RGtransConfFeaturesNonTextTest)
    FLint_pred = FLint_clf.predict(FeaturePreprocessing.FLtransIntFeaturesNonTextTest)
    CDint_pred = CDint_clf.predict(FeaturePreprocessing.CDtransIntFeaturesNonTextTest)
    RPint_pred = RPint_clf.predict(FeaturePreprocessing.RPtransIntFeaturesNonTextTest)
    RGint_pred = RGint_clf.predict(FeaturePreprocessing.RGtransIntFeaturesNonTextTest)
    FLavail_pred = FLavail_clf.predict(FeaturePreprocessing.FLtransAvailFeaturesNonTextTest)
    CDavail_pred = CDavail_clf.predict(FeaturePreprocessing.CDtransAvailFeaturesNonTextTest)
    RPavail_pred = RPavail_clf.predict(FeaturePreprocessing.RPtransAvailFeaturesNonTextTest)
    RGavail_pred = RGavail_clf.predict(FeaturePreprocessing.RGtransAvailFeaturesNonTextTest)

    nb_pred_time = round(time()-t0,3) - nb_training_time
    print 'prediction complete: ',nb_pred_time, 's'

    #compute accuracy
    classifier_acc=[]
    classifier_acc.append(accuracy_score(FLconf_pred, FeaturePreprocessing.transLabelsTest[0]))
    classifier_acc.append(accuracy_score(CDconf_pred, FeaturePreprocessing.transLabelsTest[1]))
    classifier_acc.append(accuracy_score(RPconf_pred, FeaturePreprocessing.transLabelsTest[2]))
    classifier_acc.append(accuracy_score(RGconf_pred, FeaturePreprocessing.transLabelsTest[3]))
    classifier_acc.append(accuracy_score(FLint_pred, FeaturePreprocessing.transLabelsTest[4]))
    classifier_acc.append(accuracy_score(CDint_pred, FeaturePreprocessing.transLabelsTest[5]))
    classifier_acc.append(accuracy_score(RPint_pred, FeaturePreprocessing.transLabelsTest[6]))
    classifier_acc.append(accuracy_score(RGint_pred, FeaturePreprocessing.transLabelsTest[7]))
    classifier_acc.append(accuracy_score(FLavail_pred, FeaturePreprocessing.transLabelsTest[8]))
    classifier_acc.append(accuracy_score(CDavail_pred, FeaturePreprocessing.transLabelsTest[9]))
    classifier_acc.append(accuracy_score(RPavail_pred, FeaturePreprocessing.transLabelsTest[10]))
    classifier_acc.append(accuracy_score(RGavail_pred, FeaturePreprocessing.transLabelsTest[11]))

    print 'FL confidentiality prediction accuracy: ', classifier_acc[0]
    print 'CD confidentiality prediction accuracy: ', classifier_acc[1]
    print 'RP confidentiality prediction accuracy: ', classifier_acc[2]
    print 'RG confidentiality prediction accuracy: ', classifier_acc[3]
    print 'FL integrity prediction accuracy: ', classifier_acc[4]
    print 'FL integrity prediction accuracy: ', classifier_acc[5]
    print 'RP integrity prediction accuracy: ', classifier_acc[6]
    print 'RG integrity prediction accuracy: ', classifier_acc[7]
    print 'FL availability prediction accuracy: ', classifier_acc[8]
    print 'CD availability prediction accuracy: ', classifier_acc[9]
    print 'RP availability prediction accuracy: ', classifier_acc[10]
    print 'RG availability prediction accuracy: ', classifier_acc[11]
    print 'Overall accuracy: ', np.mean(classifier_acc)

    if save:
        print 'saving training classifier data...'

        from sklearn.externals import joblib
        directory = str(classifier)
        directory += '_'
        directory += str(int(np.mean(classifier_acc)*100))
        directory += '%acc_TrainingSet_'
        directory += str(len(FeaturePreprocessing.FeatureExtraction.training_dict_list))
        directory += '_'
        directory += str(datetime.date.today())


        #make directory if one doesnt exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        #filename
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
        outputMetrics = directory +str('/metrics.txt')

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

        text_file = open(outputMetrics, "w+")
        text_file.write('FL confidentiality prediction accuracy: %.7f \n' % classifier_acc[0])
        text_file.write('CD confidentiality prediction accuracy: %.7f \n' % classifier_acc[1])
        text_file.write('RP confidentiality prediction accuracy: %.7f \n' % classifier_acc[2])
        text_file.write('RG confidentiality prediction accuracy: %.7f \n' % classifier_acc[3])
        text_file.write('FL integrity prediction accuracy: %.7f \n' % classifier_acc[4])
        text_file.write('FL integrity prediction accuracy: %.7f \n' % classifier_acc[5])
        text_file.write('RP integrity prediction accuracy: %.7f \n' % classifier_acc[6])
        text_file.write('RG integrity prediction accuracy: %.7f \n' % classifier_acc[7])
        text_file.write('FL availability prediction accuracy: %.7f \n' % classifier_acc[8])
        text_file.write('CD availability prediction accuracy: %.7f \n' % classifier_acc[9])
        text_file.write('RP availability prediction accuracy: %.7f \n' % classifier_acc[10])
        text_file.write('RG availability prediction accuracy: %.7f \n' % classifier_acc[11])
        text_file.write('Overall accuracy: %.7f \n' % np.mean(classifier_acc))
        text_file.close()
