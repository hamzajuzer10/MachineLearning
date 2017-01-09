#!/usr/bin/python

#Prepocess labels, non-text and text features

import FeatureActualExtraction
import constants
from PreprocessingFunctions import label_transform, label_inv_transform, nonTextFeature_transform, nonTextFeature_nvalues, checkLabelsNFeatures
from sklearn.preprocessing import OneHotEncoder


#Features (non-text)

print 'transforming actual features (non-text)...'
#train
FLfitConfFeaturesNonTextActual = nonTextFeature_transform(FeatureActualExtraction.FLconfidentialityNonTextActualFeatures)
CDfitConfFeaturesNonTextActual = nonTextFeature_transform(FeatureActualExtraction.CDconfidentialityNonTextActualFeatures)
RPfitConfFeaturesNonTextActual = nonTextFeature_transform(FeatureActualExtraction.RPconfidentialityNonTextActualFeatures)
RGfitConfFeaturesNonTextActual = nonTextFeature_transform(FeatureActualExtraction.RGconfidentialityNonTextActualFeatures)
FLfitIntFeaturesNonTextActual = nonTextFeature_transform(FeatureActualExtraction.FLintegrityNonTextActualFeatures)
CDfitIntFeaturesNonTextActual = nonTextFeature_transform(FeatureActualExtraction.CDintegrityNonTextActualFeatures)
RPfitIntFeaturesNonTextActual = nonTextFeature_transform(FeatureActualExtraction.RPintegrityNonTextActualFeatures)
RGfitIntFeaturesNonTextActual = nonTextFeature_transform(FeatureActualExtraction.RGintegrityNonTextActualFeatures)
FLfitAvailFeaturesNonTextActual = nonTextFeature_transform(FeatureActualExtraction.FLavailabilityNonTextActualFeatures)
CDfitAvailFeaturesNonTextActual = nonTextFeature_transform(FeatureActualExtraction.CDavailabilityNonTextActualFeatures)
RPfitAvailFeaturesNonTextActual = nonTextFeature_transform(FeatureActualExtraction.RPavailabilityNonTextActualFeatures)
RGfitAvailFeaturesNonTextActual = nonTextFeature_transform(FeatureActualExtraction.RGavailabilityNonTextActualFeatures)

print 'loading encoder model...'

# filename (encoders)
en_1 = constants.directory + str('/FLconfEncoder.pkl')
en_2 = constants.directory + str('/CDconfEncoder.pkl')
en_3 = constants.directory + str('/RPconfEncoder.pkl')
en_4 = constants.directory + str('/RGconfEncoder.pkl')
en_5 = constants.directory + str('/FLintEncoder.pkl')
en_6 = constants.directory + str('/CDintEncoder.pkl')
en_7 = constants.directory + str('/RPintEncoder.pkl')
en_8 = constants.directory + str('/RGintEncoder.pkl')
en_9 = constants.directory + str('/FLavailEncoder.pkl')
en_10 = constants.directory + str('/CDavailEncoder.pkl')
en_11 = constants.directory + str('/RPavailEncoder.pkl')
en_12 = constants.directory + str('/RGavailEncoder.pkl')

from sklearn.externals import joblib


#one-hot encoding
FLconf_enc = joblib.load(en_1)
CDconf_enc = joblib.load(en_2)
RPconf_enc = joblib.load(en_3)
RGconf_enc = joblib.load(en_4)
FLint_enc = joblib.load(en_5)
CDint_enc = joblib.load(en_6)
RPint_enc = joblib.load(en_7)
RGint_enc = joblib.load(en_8)
FLavail_enc = joblib.load(en_9)
CDavail_enc = joblib.load(en_10)
RPavail_enc = joblib.load(en_11)
RGavail_enc = joblib.load(en_12)

print 'preprocessing actual features (non-text)...'

#transform actual values
FLtransConfFeaturesNonTextActual = FLconf_enc.transform(FLfitConfFeaturesNonTextActual).toarray()
CDtransConfFeaturesNonTextActual = CDconf_enc.transform(CDfitConfFeaturesNonTextActual).toarray()
RPtransConfFeaturesNonTextActual = RPconf_enc.transform(RPfitConfFeaturesNonTextActual).toarray()
RGtransConfFeaturesNonTextActual = RGconf_enc.transform(RGfitConfFeaturesNonTextActual).toarray()
FLtransIntFeaturesNonTextActual = FLint_enc.transform(FLfitIntFeaturesNonTextActual).toarray()
CDtransIntFeaturesNonTextActual = CDint_enc.transform(CDfitIntFeaturesNonTextActual).toarray()
RPtransIntFeaturesNonTextActual = RPint_enc.transform(RPfitIntFeaturesNonTextActual).toarray()
RGtransIntFeaturesNonTextActual = RGint_enc.transform(RGfitIntFeaturesNonTextActual).toarray()
FLtransAvailFeaturesNonTextActual = FLavail_enc.transform(FLfitAvailFeaturesNonTextActual).toarray()
CDtransAvailFeaturesNonTextActual = CDavail_enc.transform(CDfitAvailFeaturesNonTextActual).toarray()
RPtransAvailFeaturesNonTextActual = RPavail_enc.transform(RPfitAvailFeaturesNonTextActual).toarray()
RGtransAvailFeaturesNonTextActual = RGavail_enc.transform(RGfitAvailFeaturesNonTextActual).toarray()


# #feature selection, to reduce overall dimensionality and computational time - TODO


#Features - text - TODO
