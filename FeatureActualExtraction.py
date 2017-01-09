#!/usr/bin/python

#Feature extraction for the extracted Excel data
#We will have 3 separate list of features, one for C,I,A
#We will first extract the non-text features and then append the text features

import random
import sys
import constants

import ExcelActualExtraction

# Actual features (non-text)
FLconfidentialityNonTextActualFeatures = []
CDconfidentialityNonTextActualFeatures = []
RPconfidentialityNonTextActualFeatures = []
RGconfidentialityNonTextActualFeatures = []
FLintegrityNonTextActualFeatures = []
CDintegrityNonTextActualFeatures = []
RPintegrityNonTextActualFeatures = []
RGintegrityNonTextActualFeatures = []
FLavailabilityNonTextActualFeatures = []
CDavailabilityNonTextActualFeatures = []
RPavailabilityNonTextActualFeatures = []
RGavailabilityNonTextActualFeatures = []

actual_dict_list = ExcelActualExtraction.dict_list
print 'No. of actual data samples: ', len(actual_dict_list)

# Extract actual features(non-text)
# Ensure that features are cleansed and missing values are imputed - TODO

print 'loading training features...'

# check if directory containing trained classifier is specified
if constants.directory == None:
    print 'no directory for trained classifier selected...'
    print 'please enter a trained classifier directory...'
    print 'program is exiting...'
    sys.exit(0)

tf = constants.directory + str('/nNonTextFeature.pkl')

from sklearn.externals import joblib
nNonText_feature = joblib.load(tf)

print 'extracting training and text features (non-text)...'

for item in actual_dict_list:

    FLCtemp = []
    CDCtemp = []
    RPCtemp = []
    RGCtemp = []
    FLItemp = []
    CDItemp = []
    RPItemp = []
    RGItemp = []
    FLAtemp = []
    CDAtemp = []
    RPAtemp = []
    RGAtemp = []

    for feature in nNonText_feature['FLconfidentialityNonText_features']:
        FLCtemp.append(item[feature])
    for feature in nNonText_feature['CDconfidentialityNonText_features']:
        CDCtemp.append(item[feature])
    for feature in nNonText_feature['RPconfidentialityNonText_features']:
        RPCtemp.append(item[feature])
    for feature in nNonText_feature['RGconfidentialityNonText_features']:
        RGCtemp.append(item[feature])
    for feature in nNonText_feature['FLintegrityNonText_features']:
        FLItemp.append(item[feature])
    for feature in nNonText_feature['CDintegrityNonText_features']:
        CDItemp.append(item[feature])
    for feature in nNonText_feature['RPintegrityNonText_features']:
        RPItemp.append(item[feature])
    for feature in nNonText_feature['RGintegrityNonText_features']:
        RGItemp.append(item[feature])
    for feature in nNonText_feature['FLavailabilityNonText_features']:
        FLAtemp.append(item[feature])
    for feature in nNonText_feature['CDavailabilityNonText_features']:
        CDAtemp.append(item[feature])
    for feature in nNonText_feature['RPavailabilityNonText_features']:
        RPAtemp.append(item[feature])
    for feature in nNonText_feature['RGavailabilityNonText_features']:
        RGAtemp.append(item[feature])

    FLconfidentialityNonTextActualFeatures.append(FLCtemp)
    CDconfidentialityNonTextActualFeatures.append(CDCtemp)
    RPconfidentialityNonTextActualFeatures.append(RPCtemp)
    RGconfidentialityNonTextActualFeatures.append(RGCtemp)
    FLintegrityNonTextActualFeatures.append(FLItemp)
    CDintegrityNonTextActualFeatures.append(CDItemp)
    RPintegrityNonTextActualFeatures.append(RPItemp)
    RGintegrityNonTextActualFeatures.append(RGItemp)
    FLavailabilityNonTextActualFeatures.append(FLAtemp)
    CDavailabilityNonTextActualFeatures.append(CDAtemp)
    RPavailabilityNonTextActualFeatures.append(RPAtemp)
    RGavailabilityNonTextActualFeatures.append(RGAtemp)

# Extract actual features (text) - TODO