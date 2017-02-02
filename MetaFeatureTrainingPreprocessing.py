#!/usr/bin/python

#Prepocess labels, non-text and text features

import constants
from PreprocessingFunctions import label_transform, metaNonTextFeature_combine, nonTextFeature_transform, nonTextFeature_nvalues, checkLabelsNFeatures
from sklearn.preprocessing import OneHotEncoder

def MetaFeaturePreprocess(mconf_pred, mconf_trpred):

    #Labels

    print 'transforming meta training and test labels...'
    #train and test labels
    MetatransLabelsTrain = label_transform(FeatureTrainingExtraction.trainingLabels)
    MetatransLabelsTest = label_transform(FeatureTrainingExtraction.testLabels)

    #Features (non-text)


    print 'transforming meta training and test features (non-text)...'
    #train
    MetaFLfitConfFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.MetaFLconfidentialityNonTextTrainingFeatures)
    MetaCDfitConfFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.MetaCDconfidentialityNonTextTrainingFeatures)
    MetaRPfitConfFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.MetaRPconfidentialityNonTextTrainingFeatures)
    MetaRGfitConfFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.MetaRGconfidentialityNonTextTrainingFeatures)
    MetaFLfitIntFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.MetaFLintegrityNonTextTrainingFeatures)
    MetaCDfitIntFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.MetaCDintegrityNonTextTrainingFeatures)
    MetaRPfitIntFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.MetaRPintegrityNonTextTrainingFeatures)
    MetaRGfitIntFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.MetaRGintegrityNonTextTrainingFeatures)
    MetaFLfitAvailFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.MetaFLavailabilityNonTextTrainingFeatures)
    MetaCDfitAvailFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.MetaCDavailabilityNonTextTrainingFeatures)
    MetaRPfitAvailFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.MetaRPavailabilityNonTextTrainingFeatures)
    MetaRGfitAvailFeaturesNonTextTrain = nonTextFeature_transform(FeatureTrainingExtraction.MetaRGavailabilityNonTextTrainingFeatures)

    #test
    MetaFLfitConfFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.MetaFLconfidentialityNonTextTestFeatures)
    MetaCDfitConfFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.MetaCDconfidentialityNonTextTestFeatures)
    MetaRPfitConfFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.MetaRPconfidentialityNonTextTestFeatures)
    MetaRGfitConfFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.MetaRGconfidentialityNonTextTestFeatures)
    MetaFLfitIntFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.MetaFLintegrityNonTextTestFeatures)
    MetaCDfitIntFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.MetaCDintegrityNonTextTestFeatures)
    MetaRPfitIntFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.MetaRPintegrityNonTextTestFeatures)
    MetaRGfitIntFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.MetaRGintegrityNonTextTestFeatures)
    MetaFLfitAvailFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.MetaFLavailabilityNonTextTestFeatures)
    MetaCDfitAvailFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.MetaCDavailabilityNonTextTestFeatures)
    MetaRPfitAvailFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.MetaRPavailabilityNonTextTestFeatures)
    MetaRGfitAvailFeaturesNonTextTest = nonTextFeature_transform(FeatureTrainingExtraction.MetaRGavailabilityNonTextTestFeatures)

    #add predicted test and train labels as features
    #train
    MetaFLfitConfFeaturesNonTextTrain = metaNonTextFeature_combine(MetaFLfitConfFeaturesNonTextTrain,mconf_pred,'FLconfidentialityNonText_features')
    MetaCDfitConfFeaturesNonTextTrain = metaNonTextFeature_combine(MetaCDfitConfFeaturesNonTextTrain,mconf_pred,'CDconfidentialityNonText_features')
    MetaRPfitConfFeaturesNonTextTrain = metaNonTextFeature_combine(MetaRPfitConfFeaturesNonTextTrain,mconf_pred,'RPconfidentialityNonText_features')
    MetaRGfitConfFeaturesNonTextTrain = metaNonTextFeature_combine(MetaRGfitConfFeaturesNonTextTrain,mconf_pred,'RGconfidentialityNonText_features')
    MetaFLfitIntFeaturesNonTextTrain = metaNonTextFeature_combine(MetaFLfitIntFeaturesNonTextTrain,mconf_pred,'FLintegrityNonText_features')
    MetaCDfitIntFeaturesNonTextTrain = metaNonTextFeature_combine(MetaCDfitIntFeaturesNonTextTrain,mconf_pred,'CDintegrityNonText_features')
    MetaRPfitIntFeaturesNonTextTrain = metaNonTextFeature_combine(MetaRPfitIntFeaturesNonTextTrain,mconf_pred,'RPintegrityNonText_features')
    MetaRGfitIntFeaturesNonTextTrain = metaNonTextFeature_combine(MetaRGfitIntFeaturesNonTextTrain,mconf_pred,'RGintegrityNonText_features')
    MetaFLfitAvailFeaturesNonTextTrain = metaNonTextFeature_combine(MetaFLfitAvailFeaturesNonTextTrain,mconf_pred,'FLavailabilityNonText_features')
    MetaCDfitAvailFeaturesNonTextTrain = metaNonTextFeature_combine(MetaCDfitAvailFeaturesNonTextTrain,mconf_pred,'CDavailabilityNonText_features')
    MetaRPfitAvailFeaturesNonTextTrain = metaNonTextFeature_combine(MetaRPfitAvailFeaturesNonTextTrain,mconf_pred,'RPavailabilityNonText_features')
    MetaRGfitAvailFeaturesNonTextTrain = metaNonTextFeature_combine(MetaRGfitAvailFeaturesNonTextTrain,mconf_pred,'RGavailabilityNonText_features')

    #test

    MetaFLfitConfFeaturesNonTextTest = metaNonTextFeature_combine(MetaFLfitConfFeaturesNonTextTest,mconf_pred,'FLconfidentialityNonText_features')
    MetaCDfitConfFeaturesNonTextTest = metaNonTextFeature_combine(MetaCDfitConfFeaturesNonTextTest,mconf_pred,'CDconfidentialityNonText_features')
    MetaRPfitConfFeaturesNonTextTest = metaNonTextFeature_combine(MetaRPfitConfFeaturesNonTextTest,mconf_pred,'RPconfidentialityNonText_features')
    MetaRGfitConfFeaturesNonTextTest = metaNonTextFeature_combine(MetaRGfitConfFeaturesNonTextTest,mconf_pred,'RGconfidentialityNonText_features')
    MetaFLfitIntFeaturesNonTextTest = metaNonTextFeature_combine(MetaFLfitIntFeaturesNonTextTest,mconf_pred,'FLintegrityNonText_features')
    MetaCDfitIntFeaturesNonTextTest = metaNonTextFeature_combine(MetaCDfitIntFeaturesNonTextTest,mconf_pred,'CDintegrityNonText_features')
    MetaRPfitIntFeaturesNonTextTest = metaNonTextFeature_combine(MetaRPfitIntFeaturesNonTextTest,mconf_pred,'RPintegrityNonText_features')
    MetaRGfitIntFeaturesNonTextTest = metaNonTextFeature_combine(MetaRGfitIntFeaturesNonTextTest,mconf_pred,'RGintegrityNonText_features')
    MetaFLfitAvailFeaturesNonTextTest = metaNonTextFeature_combine(MetaFLfitAvailFeaturesNonTextTest,mconf_pred,'FLavailabilityNonText_features')
    MetaCDfitAvailFeaturesNonTextTest = metaNonTextFeature_combine(MetaCDfitAvailFeaturesNonTextTest,mconf_pred,'CDavailabilityNonText_features')
    MetaRPfitAvailFeaturesNonTextTest = metaNonTextFeature_combine(MetaRPfitAvailFeaturesNonTextTest,mconf_pred,'RPavailabilityNonText_features')
    MetaRGfitAvailFeaturesNonTextTest = metaNonTextFeature_combine(MetaRGfitAvailFeaturesNonTextTest,mconf_pred,'RGavailabilityNonText_features')

    #determine no. of unique values for each feature

    #combine the Base and Meta features
    MetaFLconf_features = constants.nMetaBaseNonText_feature['FLconfidentialityNonText_features'] + constants.nMetaMetaNonText_feature['FLconfidentialityNonText_features']
    MetaCDconf_features = constants.nMetaBaseNonText_feature['CDconfidentialityNonText_features'] + constants.nMetaMetaNonText_feature['CDconfidentialityNonText_features']
    MetaRPconf_features = constants.nMetaBaseNonText_feature['RPconfidentialityNonText_features'] + constants.nMetaMetaNonText_feature['RPconfidentialityNonText_features']
    MetaRGconf_features = constants.nMetaBaseNonText_feature['RGconfidentialityNonText_features'] + constants.nMetaMetaNonText_feature['RGconfidentialityNonText_features']
    MetaFLint_features = constants.nMetaBaseNonText_feature['FLintegrityNonText_features'] + constants.nMetaMetaNonText_feature['FLintegrityNonText_features']
    MetaCDint_features = constants.nMetaBaseNonText_feature['CDintegrityNonText_features'] + constants.nMetaMetaNonText_feature['CDintegrityNonText_features']
    MetaRPint_features = constants.nMetaBaseNonText_feature['RPintegrityNonText_features'] + constants.nMetaMetaNonText_feature['RPintegrityNonText_features']
    MetaRGint_features = constants.nMetaBaseNonText_feature['RGintegrityNonText_features'] + constants.nMetaMetaNonText_feature['RGintegrityNonText_features']
    MetaFLavail_features = constants.nMetaBaseNonText_feature['FLavailabilityNonText_features'] + constants.nMetaMetaNonText_feature['FLavailabilityNonText_features']
    MetaCDavail_features = constants.nMetaBaseNonText_feature['CDavailabilityNonText_features'] + constants.nMetaMetaNonText_feature['CDavailabilityNonText_features']
    MetaRPavail_features = constants.nMetaBaseNonText_feature['RPavailabilityNonText_features'] + constants.nMetaMetaNonText_feature['RPavailabilityNonText_features']
    MetaRGavail_features = constants.nMetaBaseNonText_feature['RGavailabilityNonText_features'] + constants.nMetaMetaNonText_feature['RGavailabilityNonText_features']



    #determine no. of unique values for each feature
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


    #one-hot encoding
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
    #fit train values
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

    #transform train values
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

    #transform test values
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


    #verify correct labels and features
    print 'verifying accuracy of labels and feature matrices...'
    checkLabelsNFeatures(MetaFLtransConfFeaturesNonTextTrain,MetatransLabelsTrain[0],'FL confidentiality training')
    checkLabelsNFeatures(MetaCDtransConfFeaturesNonTextTrain,MetatransLabelsTrain[1],'CD confidentiality training')
    checkLabelsNFeatures(MetaRPtransConfFeaturesNonTextTrain,MetatransLabelsTrain[2],'RP confidentiality training')
    checkLabelsNFeatures(MetaRGtransConfFeaturesNonTextTrain,MetatransLabelsTrain[3],'RG confidentiality training')
    checkLabelsNFeatures(MetaFLtransIntFeaturesNonTextTrain,MetatransLabelsTrain[4],'FL integrity training')
    checkLabelsNFeatures(MetaCDtransIntFeaturesNonTextTrain,MetatransLabelsTrain[5],'CD integrity training')
    checkLabelsNFeatures(MetaRPtransIntFeaturesNonTextTrain,MetatransLabelsTrain[6],'RP integrity training')
    checkLabelsNFeatures(MetaRGtransIntFeaturesNonTextTrain,MetatransLabelsTrain[7],'RG integrity training')
    checkLabelsNFeatures(MetaFLtransAvailFeaturesNonTextTrain,MetatransLabelsTrain[8],'FL availability training')
    checkLabelsNFeatures(MetaCDtransAvailFeaturesNonTextTrain,MetatransLabelsTrain[9],'CD availability training')
    checkLabelsNFeatures(MetaRPtransAvailFeaturesNonTextTrain,MetatransLabelsTrain[10],'RP availability training')
    checkLabelsNFeatures(MetaRGtransAvailFeaturesNonTextTrain,MetatransLabelsTrain[11],'RG availability training')

    if constants.train_test_split != 1.0:
        checkLabelsNFeatures(MetaFLtransConfFeaturesNonTextTest,MetatransLabelsTest[0],'FL confidentiality testing')
        checkLabelsNFeatures(MetaCDtransConfFeaturesNonTextTest,MetatransLabelsTest[1],'CD confidentiality testing')
        checkLabelsNFeatures(MetaRPtransConfFeaturesNonTextTest,MetatransLabelsTest[2],'RP confidentiality testing')
        checkLabelsNFeatures(MetaRGtransConfFeaturesNonTextTest,MetatransLabelsTest[3],'RG confidentiality testing')
        checkLabelsNFeatures(MetaFLtransIntFeaturesNonTextTest,MetatransLabelsTest[4],'FL integrity testing')
        checkLabelsNFeatures(MetaCDtransIntFeaturesNonTextTest,MetatransLabelsTest[5],'CD integrity testing')
        checkLabelsNFeatures(MetaRPtransIntFeaturesNonTextTest,MetatransLabelsTest[6],'RP integrity testing')
        checkLabelsNFeatures(MetaRGtransIntFeaturesNonTextTest,MetatransLabelsTest[7],'RG integrity testing')
        checkLabelsNFeatures(MetaFLtransAvailFeaturesNonTextTest,MetatransLabelsTest[8],'FL availability testing')
        checkLabelsNFeatures(MetaCDtransAvailFeaturesNonTextTest,MetatransLabelsTest[9],'CD availability testing')
        checkLabelsNFeatures(MetaRPtransAvailFeaturesNonTextTest,MetatransLabelsTest[10],'RP availability testing')
        checkLabelsNFeatures(MetaRGtransAvailFeaturesNonTextTest,MetatransLabelsTest[11],'RG availability testing')

