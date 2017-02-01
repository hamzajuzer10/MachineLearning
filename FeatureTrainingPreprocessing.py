#!/usr/bin/python

#Prepocess labels, non-text and text features

import constants
import FeatureTrainingExtraction
from PreprocessingFunctions import label_transform, label_inv_transform, nonTextFeature_transform, nonTextFeature_nvalues, checkLabelsNFeatures
from sklearn.preprocessing import OneHotEncoder

#Labels

print 'transforming training and test labels...'
#train and test labels
transLabelsTrain = label_transform(FeatureTrainingExtraction.trainingLabels)
transLabelsTest = label_transform(FeatureTrainingExtraction.testLabels)

#Features (non-text)

print 'transforming training and test features (non-text)...'
#train
FLfitConfFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.FLconfidentialityNonTextTrainingFeatures)
CDfitConfFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.CDconfidentialityNonTextTrainingFeatures)
RPfitConfFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.RPconfidentialityNonTextTrainingFeatures)
RGfitConfFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.RGconfidentialityNonTextTrainingFeatures)
FLfitIntFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.FLintegrityNonTextTrainingFeatures)
CDfitIntFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.CDintegrityNonTextTrainingFeatures)
RPfitIntFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.RPintegrityNonTextTrainingFeatures)
RGfitIntFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.RGintegrityNonTextTrainingFeatures)
FLfitAvailFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.FLavailabilityNonTextTrainingFeatures)
CDfitAvailFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.CDavailabilityNonTextTrainingFeatures)
RPfitAvailFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.RPavailabilityNonTextTrainingFeatures)
RGfitAvailFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.RGavailabilityNonTextTrainingFeatures)

#test
FLfitConfFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.FLconfidentialityNonTextTestFeatures)
CDfitConfFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.CDconfidentialityNonTextTestFeatures)
RPfitConfFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.RPconfidentialityNonTextTestFeatures)
RGfitConfFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.RGconfidentialityNonTextTestFeatures)
FLfitIntFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.FLintegrityNonTextTestFeatures)
CDfitIntFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.CDintegrityNonTextTestFeatures)
RPfitIntFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.RPintegrityNonTextTestFeatures)
RGfitIntFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.RGintegrityNonTextTestFeatures)
FLfitAvailFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.FLavailabilityNonTextTestFeatures)
CDfitAvailFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.CDavailabilityNonTextTestFeatures)
RPfitAvailFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.RPavailabilityNonTextTestFeatures)
RGfitAvailFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.RGavailabilityNonTextTestFeatures)

#determine no. of unique values for each feature
FLconf_nvalues = nonTextFeature_nvalues(constants.nNonText_feature['FLconfidentialityNonText_features'])
CDconf_nvalues = nonTextFeature_nvalues(constants.nNonText_feature['CDconfidentialityNonText_features'])
RPconf_nvalues = nonTextFeature_nvalues(constants.nNonText_feature['RPconfidentialityNonText_features'])
RGconf_nvalues = nonTextFeature_nvalues(constants.nNonText_feature['RGconfidentialityNonText_features'])
FLint_nvalues = nonTextFeature_nvalues(constants.nNonText_feature['FLintegrityNonText_features'])
CDint_nvalues = nonTextFeature_nvalues(constants.nNonText_feature['CDintegrityNonText_features'])
RPint_nvalues = nonTextFeature_nvalues(constants.nNonText_feature['RPintegrityNonText_features'])
RGint_nvalues = nonTextFeature_nvalues(constants.nNonText_feature['RGintegrityNonText_features'])
FLavail_nvalues = nonTextFeature_nvalues(constants.nNonText_feature['FLavailabilityNonText_features'])
CDavail_nvalues = nonTextFeature_nvalues(constants.nNonText_feature['CDavailabilityNonText_features'])
RPavail_nvalues = nonTextFeature_nvalues(constants.nNonText_feature['RPavailabilityNonText_features'])
RGavail_nvalues = nonTextFeature_nvalues(constants.nNonText_feature['RGavailabilityNonText_features'])


#one-hot encoding
FLconf_enc = OneHotEncoder(n_values=FLconf_nvalues)
CDconf_enc = OneHotEncoder(n_values=CDconf_nvalues)
RPconf_enc = OneHotEncoder(n_values=RPconf_nvalues)
RGconf_enc = OneHotEncoder(n_values=RGconf_nvalues)
FLint_enc = OneHotEncoder(n_values=FLint_nvalues)
CDint_enc = OneHotEncoder(n_values=CDint_nvalues)
RPint_enc = OneHotEncoder(n_values=RPint_nvalues)
RGint_enc = OneHotEncoder(n_values=RGint_nvalues)
FLavail_enc = OneHotEncoder(n_values=FLavail_nvalues)
CDavail_enc = OneHotEncoder(n_values=CDavail_nvalues)
RPavail_enc = OneHotEncoder(n_values=RPavail_nvalues)
RGavail_enc = OneHotEncoder(n_values=RGavail_nvalues)

print 'preprocessing training and test features (non-text)...'
#fit train values
FLconf_enc.fit(FLfitConfFeaturesNonTextTrain)
CDconf_enc.fit(CDfitConfFeaturesNonTextTrain)
RPconf_enc.fit(RPfitConfFeaturesNonTextTrain)
RGconf_enc.fit(RGfitConfFeaturesNonTextTrain)
FLint_enc.fit(FLfitIntFeaturesNonTextTrain)
CDint_enc.fit(CDfitIntFeaturesNonTextTrain)
RPint_enc.fit(RPfitIntFeaturesNonTextTrain)
RGint_enc.fit(RGfitIntFeaturesNonTextTrain)
FLavail_enc.fit(FLfitAvailFeaturesNonTextTrain)
CDavail_enc.fit(CDfitAvailFeaturesNonTextTrain)
RPavail_enc.fit(RPfitAvailFeaturesNonTextTrain)
RGavail_enc.fit(RGfitAvailFeaturesNonTextTrain)

#transform train values
FLtransConfFeaturesNonTextTrain = FLconf_enc.transform(FLfitConfFeaturesNonTextTrain).toarray()
CDtransConfFeaturesNonTextTrain = CDconf_enc.transform(CDfitConfFeaturesNonTextTrain).toarray()
RPtransConfFeaturesNonTextTrain = RPconf_enc.transform(RPfitConfFeaturesNonTextTrain).toarray()
RGtransConfFeaturesNonTextTrain = RGconf_enc.transform(RGfitConfFeaturesNonTextTrain).toarray()
FLtransIntFeaturesNonTextTrain = FLint_enc.transform(FLfitIntFeaturesNonTextTrain).toarray()
CDtransIntFeaturesNonTextTrain = CDint_enc.transform(CDfitIntFeaturesNonTextTrain).toarray()
RPtransIntFeaturesNonTextTrain = RPint_enc.transform(RPfitIntFeaturesNonTextTrain).toarray()
RGtransIntFeaturesNonTextTrain = RGint_enc.transform(RGfitIntFeaturesNonTextTrain).toarray()
FLtransAvailFeaturesNonTextTrain = FLavail_enc.transform(FLfitAvailFeaturesNonTextTrain).toarray()
CDtransAvailFeaturesNonTextTrain = CDavail_enc.transform(CDfitAvailFeaturesNonTextTrain).toarray()
RPtransAvailFeaturesNonTextTrain = RPavail_enc.transform(RPfitAvailFeaturesNonTextTrain).toarray()
RGtransAvailFeaturesNonTextTrain = RGavail_enc.transform(RGfitAvailFeaturesNonTextTrain).toarray()

#transform test values
if constants.train_test_split != 1.0:
    FLtransConfFeaturesNonTextTest = FLconf_enc.transform(FLfitConfFeaturesNonTextTest).toarray()
    CDtransConfFeaturesNonTextTest = CDconf_enc.transform(CDfitConfFeaturesNonTextTest).toarray()
    RPtransConfFeaturesNonTextTest = RPconf_enc.transform(RPfitConfFeaturesNonTextTest).toarray()
    RGtransConfFeaturesNonTextTest = RGconf_enc.transform(RGfitConfFeaturesNonTextTest).toarray()
    FLtransIntFeaturesNonTextTest = FLint_enc.transform(FLfitIntFeaturesNonTextTest).toarray()
    CDtransIntFeaturesNonTextTest = CDint_enc.transform(CDfitIntFeaturesNonTextTest).toarray()
    RPtransIntFeaturesNonTextTest = RPint_enc.transform(RPfitIntFeaturesNonTextTest).toarray()
    RGtransIntFeaturesNonTextTest = RGint_enc.transform(RGfitIntFeaturesNonTextTest).toarray()
    FLtransAvailFeaturesNonTextTest = FLavail_enc.transform(FLfitAvailFeaturesNonTextTest).toarray()
    CDtransAvailFeaturesNonTextTest = CDavail_enc.transform(CDfitAvailFeaturesNonTextTest).toarray()
    RPtransAvailFeaturesNonTextTest = RPavail_enc.transform(RPfitAvailFeaturesNonTextTest).toarray()
    RGtransAvailFeaturesNonTextTest = RGavail_enc.transform(RGfitAvailFeaturesNonTextTest).toarray()

# #feature selection, to reduce overall dimensionality and computational time - TODO


#verify correct labels and features
print 'verifying accuracy of labels and feature matrices...'
checkLabelsNFeatures(FLtransConfFeaturesNonTextTrain,transLabelsTrain[0],'FL confidentiality training')
checkLabelsNFeatures(CDtransConfFeaturesNonTextTrain,transLabelsTrain[1],'CD confidentiality training')
checkLabelsNFeatures(RPtransConfFeaturesNonTextTrain,transLabelsTrain[2],'RP confidentiality training')
checkLabelsNFeatures(RGtransConfFeaturesNonTextTrain,transLabelsTrain[3],'RG confidentiality training')
checkLabelsNFeatures(FLtransIntFeaturesNonTextTrain,transLabelsTrain[4],'FL integrity training')
checkLabelsNFeatures(CDtransIntFeaturesNonTextTrain,transLabelsTrain[5],'CD integrity training')
checkLabelsNFeatures(RPtransIntFeaturesNonTextTrain,transLabelsTrain[6],'RP integrity training')
checkLabelsNFeatures(RGtransIntFeaturesNonTextTrain,transLabelsTrain[7],'RG integrity training')
checkLabelsNFeatures(FLtransAvailFeaturesNonTextTrain,transLabelsTrain[8],'FL availability training')
checkLabelsNFeatures(CDtransAvailFeaturesNonTextTrain,transLabelsTrain[9],'CD availability training')
checkLabelsNFeatures(RPtransAvailFeaturesNonTextTrain,transLabelsTrain[10],'RP availability training')
checkLabelsNFeatures(RGtransAvailFeaturesNonTextTrain,transLabelsTrain[11],'RG availability training')

if constants.train_test_split != 1.0:
    checkLabelsNFeatures(FLtransConfFeaturesNonTextTest,transLabelsTest[0],'FL confidentiality testing')
    checkLabelsNFeatures(CDtransConfFeaturesNonTextTest,transLabelsTest[1],'CD confidentiality testing')
    checkLabelsNFeatures(RPtransConfFeaturesNonTextTest,transLabelsTest[2],'RP confidentiality testing')
    checkLabelsNFeatures(RGtransConfFeaturesNonTextTest,transLabelsTest[3],'RG confidentiality testing')
    checkLabelsNFeatures(FLtransIntFeaturesNonTextTest,transLabelsTest[4],'FL integrity testing')
    checkLabelsNFeatures(CDtransIntFeaturesNonTextTest,transLabelsTest[5],'CD integrity testing')
    checkLabelsNFeatures(RPtransIntFeaturesNonTextTest,transLabelsTest[6],'RP integrity testing')
    checkLabelsNFeatures(RGtransIntFeaturesNonTextTest,transLabelsTest[7],'RG integrity testing')
    checkLabelsNFeatures(FLtransAvailFeaturesNonTextTest,transLabelsTest[8],'FL availability testing')
    checkLabelsNFeatures(CDtransAvailFeaturesNonTextTest,transLabelsTest[9],'CD availability testing')
    checkLabelsNFeatures(RPtransAvailFeaturesNonTextTest,transLabelsTest[10],'RP availability testing')
    checkLabelsNFeatures(RGtransAvailFeaturesNonTextTest,transLabelsTest[11],'RG availability testing')
