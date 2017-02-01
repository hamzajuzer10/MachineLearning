#!/usr/bin/python

#Feature extraction for the extracted Excel data
#We will have 3 separate list of features, one for C,I,A
#We will first extract the non-text features and then append the text features

import random
import sys
import constants
import ExcelTrainingExtraction

#Training and test labels - 12 categories
trainingLabels=[]
testLabels=[]


#Training and test features (non-text)
FLconfidentialityNonTextTrainingFeatures=[]
CDconfidentialityNonTextTrainingFeatures=[]
RPconfidentialityNonTextTrainingFeatures=[]
RGconfidentialityNonTextTrainingFeatures=[]
FLintegrityNonTextTrainingFeatures=[]
CDintegrityNonTextTrainingFeatures=[]
RPintegrityNonTextTrainingFeatures=[]
RGintegrityNonTextTrainingFeatures=[]
FLavailabilityNonTextTrainingFeatures=[]
CDavailabilityNonTextTrainingFeatures=[]
RPavailabilityNonTextTrainingFeatures=[]
RGavailabilityNonTextTrainingFeatures=[]

MetaFLconfidentialityNonTextTrainingFeatures=[]
MetaCDconfidentialityNonTextTrainingFeatures=[]
MetaRPconfidentialityNonTextTrainingFeatures=[]
MetaRGconfidentialityNonTextTrainingFeatures=[]
MetaFLintegrityNonTextTrainingFeatures=[]
MetaCDintegrityNonTextTrainingFeatures=[]
MetaRPintegrityNonTextTrainingFeatures=[]
MetaRGintegrityNonTextTrainingFeatures=[]
MetaFLavailabilityNonTextTrainingFeatures=[]
MetaCDavailabilityNonTextTrainingFeatures=[]
MetaRPavailabilityNonTextTrainingFeatures=[]
MetaRGavailabilityNonTextTrainingFeatures=[]

FLconfidentialityNonTextTestFeatures=[]
CDconfidentialityNonTextTestFeatures=[]
RPconfidentialityNonTextTestFeatures=[]
RGconfidentialityNonTextTestFeatures=[]
FLintegrityNonTextTestFeatures=[]
CDintegrityNonTextTestFeatures=[]
RPintegrityNonTextTestFeatures=[]
RGintegrityNonTextTestFeatures=[]
FLavailabilityNonTextTestFeatures=[]
CDavailabilityNonTextTestFeatures=[]
RPavailabilityNonTextTestFeatures=[]
RGavailabilityNonTextTestFeatures=[]

MetaFLconfidentialityNonTextTestFeatures=[]
MetaCDconfidentialityNonTextTestFeatures=[]
MetaRPconfidentialityNonTextTestFeatures=[]
MetaRGconfidentialityNonTextTestFeatures=[]
MetaFLintegrityNonTextTestFeatures=[]
MetaCDintegrityNonTextTestFeatures=[]
MetaRPintegrityNonTextTestFeatures=[]
MetaRGintegrityNonTextTestFeatures=[]
MetaFLavailabilityNonTextTestFeatures=[]
MetaCDavailabilityNonTextTestFeatures=[]
MetaRPavailabilityNonTextTestFeatures=[]
MetaRGavailabilityNonTextTestFeatures=[]

#Separate training and test features/ labels
#e.g. Use 95% of data as training and 5% as test

print 'separating classified training and test data...'

random.shuffle(ExcelTrainingExtraction.dict_list)
training_dict_list = ExcelTrainingExtraction.dict_list[:int(len(ExcelTrainingExtraction.dict_list)*constants.train_test_split)]
print 'No. of training data samples: ', len(training_dict_list)

test_dict_list = ExcelTrainingExtraction.dict_list[int(len(ExcelTrainingExtraction.dict_list)*constants.train_test_split):]
print 'No. of test data samples: ', len(test_dict_list)

#Extract training and test labels

print 'extracting training and test labels...'

for label in constants.labels:
    train=[]
    test=[]

    for item in training_dict_list:
        train.append(item[label])

    for item in test_dict_list:
        test.append(item[label])

    trainingLabels.append(train)
    testLabels.append(test)

#Extract training and test features(non-text)

print 'extracting training and text features (non-text)...'

for item in training_dict_list:

    FLCtemp=[]
    CDCtemp=[]
    RPCtemp=[]
    RGCtemp=[]
    FLItemp=[]
    CDItemp=[]
    RPItemp=[]
    RGItemp=[]
    FLAtemp=[]
    CDAtemp=[]
    RPAtemp=[]
    RGAtemp=[]

    for feature in constants.nNonText_feature['FLconfidentialityNonText_features']:
        FLCtemp.append(item[feature])
    for feature in constants.nNonText_feature['CDconfidentialityNonText_features']:
        CDCtemp.append(item[feature])
    for feature in constants.nNonText_feature['RPconfidentialityNonText_features']:
        RPCtemp.append(item[feature])
    for feature in constants.nNonText_feature['RGconfidentialityNonText_features']:
        RGCtemp.append(item[feature])
    for feature in constants.nNonText_feature['FLintegrityNonText_features']:
        FLItemp.append(item[feature])
    for feature in constants.nNonText_feature['CDintegrityNonText_features']:
        CDItemp.append(item[feature])
    for feature in constants.nNonText_feature['RPintegrityNonText_features']:
        RPItemp.append(item[feature])
    for feature in constants.nNonText_feature['RGintegrityNonText_features']:
        RGItemp.append(item[feature])
    for feature in constants.nNonText_feature['FLavailabilityNonText_features']:
        FLAtemp.append(item[feature])
    for feature in constants.nNonText_feature['CDavailabilityNonText_features']:
        CDAtemp.append(item[feature])
    for feature in constants.nNonText_feature['RPavailabilityNonText_features']:
        RPAtemp.append(item[feature])
    for feature in constants.nNonText_feature['RGavailabilityNonText_features']:
        RGAtemp.append(item[feature])

    FLconfidentialityNonTextTrainingFeatures.append(FLCtemp)
    CDconfidentialityNonTextTrainingFeatures.append(CDCtemp)
    RPconfidentialityNonTextTrainingFeatures.append(RPCtemp)
    RGconfidentialityNonTextTrainingFeatures.append(RGCtemp)
    FLintegrityNonTextTrainingFeatures.append(FLItemp)
    CDintegrityNonTextTrainingFeatures.append(CDItemp)
    RPintegrityNonTextTrainingFeatures.append(RPItemp)
    RGintegrityNonTextTrainingFeatures.append(RGItemp)
    FLavailabilityNonTextTrainingFeatures.append(FLAtemp)
    CDavailabilityNonTextTrainingFeatures.append(CDAtemp)
    RPavailabilityNonTextTrainingFeatures.append(RPAtemp)
    RGavailabilityNonTextTrainingFeatures.append(RGAtemp)

for item in test_dict_list:

    FLCtemp=[]
    CDCtemp=[]
    RPCtemp=[]
    RGCtemp=[]
    FLItemp=[]
    CDItemp=[]
    RPItemp=[]
    RGItemp=[]
    FLAtemp=[]
    CDAtemp=[]
    RPAtemp=[]
    RGAtemp=[]

    for feature in constants.nNonText_feature['FLconfidentialityNonText_features']:
        FLCtemp.append(item[feature])
    for feature in constants.nNonText_feature['CDconfidentialityNonText_features']:
        CDCtemp.append(item[feature])
    for feature in constants.nNonText_feature['RPconfidentialityNonText_features']:
        RPCtemp.append(item[feature])
    for feature in constants.nNonText_feature['RGconfidentialityNonText_features']:
        RGCtemp.append(item[feature])
    for feature in constants.nNonText_feature['FLintegrityNonText_features']:
        FLItemp.append(item[feature])
    for feature in constants.nNonText_feature['CDintegrityNonText_features']:
        CDItemp.append(item[feature])
    for feature in constants.nNonText_feature['RPintegrityNonText_features']:
        RPItemp.append(item[feature])
    for feature in constants.nNonText_feature['RGintegrityNonText_features']:
        RGItemp.append(item[feature])
    for feature in constants.nNonText_feature['FLavailabilityNonText_features']:
        FLAtemp.append(item[feature])
    for feature in constants.nNonText_feature['CDavailabilityNonText_features']:
        CDAtemp.append(item[feature])
    for feature in constants.nNonText_feature['RPavailabilityNonText_features']:
        RPAtemp.append(item[feature])
    for feature in constants.nNonText_feature['RGavailabilityNonText_features']:
        RGAtemp.append(item[feature])

    FLconfidentialityNonTextTestFeatures.append(FLCtemp)
    CDconfidentialityNonTextTestFeatures.append(CDCtemp)
    RPconfidentialityNonTextTestFeatures.append(RPCtemp)
    RGconfidentialityNonTextTestFeatures.append(RGCtemp)
    FLintegrityNonTextTestFeatures.append(FLItemp)
    CDintegrityNonTextTestFeatures.append(CDItemp)
    RPintegrityNonTextTestFeatures.append(RPItemp)
    RGintegrityNonTextTestFeatures.append(RGItemp)
    FLavailabilityNonTextTestFeatures.append(FLAtemp)
    CDavailabilityNonTextTestFeatures.append(CDAtemp)
    RPavailabilityNonTextTestFeatures.append(RPAtemp)
    RGavailabilityNonTextTestFeatures.append(RGAtemp)

#Extract training and test features for meta classifier
if constants.meta:

    for item in training_dict_list:

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

        for feature in constants.nMetaBaseNonText_feature['FLconfidentialityNonText_features']:
            FLCtemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['CDconfidentialityNonText_features']:
            CDCtemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['RPconfidentialityNonText_features']:
            RPCtemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['RGconfidentialityNonText_features']:
            RGCtemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['FLintegrityNonText_features']:
            FLItemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['CDintegrityNonText_features']:
            CDItemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['RPintegrityNonText_features']:
            RPItemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['RGintegrityNonText_features']:
            RGItemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['FLavailabilityNonText_features']:
            FLAtemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['CDavailabilityNonText_features']:
            CDAtemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['RPavailabilityNonText_features']:
            RPAtemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['RGavailabilityNonText_features']:
            RGAtemp.append(item[feature])

        MetaFLconfidentialityNonTextTrainingFeatures.append(FLCtemp)
        MetaCDconfidentialityNonTextTrainingFeatures.append(CDCtemp)
        MetaRPconfidentialityNonTextTrainingFeatures.append(RPCtemp)
        MetaRGconfidentialityNonTextTrainingFeatures.append(RGCtemp)
        MetaFLintegrityNonTextTrainingFeatures.append(FLItemp)
        MetaCDintegrityNonTextTrainingFeatures.append(CDItemp)
        MetaRPintegrityNonTextTrainingFeatures.append(RPItemp)
        MetaRGintegrityNonTextTrainingFeatures.append(RGItemp)
        MetaFLavailabilityNonTextTrainingFeatures.append(FLAtemp)
        MetaCDavailabilityNonTextTrainingFeatures.append(CDAtemp)
        MetaRPavailabilityNonTextTrainingFeatures.append(RPAtemp)
        MetaRGavailabilityNonTextTrainingFeatures.append(RGAtemp)

    for item in test_dict_list:

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

        for feature in constants.nMetaBaseNonText_feature['FLconfidentialityNonText_features']:
            FLCtemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['CDconfidentialityNonText_features']:
            CDCtemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['RPconfidentialityNonText_features']:
            RPCtemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['RGconfidentialityNonText_features']:
            RGCtemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['FLintegrityNonText_features']:
            FLItemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['CDintegrityNonText_features']:
            CDItemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['RPintegrityNonText_features']:
            RPItemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['RGintegrityNonText_features']:
            RGItemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['FLavailabilityNonText_features']:
            FLAtemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['CDavailabilityNonText_features']:
            CDAtemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['RPavailabilityNonText_features']:
            RPAtemp.append(item[feature])
        for feature in constants.nMetaBaseNonText_feature['RGavailabilityNonText_features']:
            RGAtemp.append(item[feature])

        MetaFLconfidentialityNonTextTestFeatures.append(FLCtemp)
        MetaCDconfidentialityNonTextTestFeatures.append(CDCtemp)
        MetaRPconfidentialityNonTextTestFeatures.append(RPCtemp)
        MetaRGconfidentialityNonTextTestFeatures.append(RGCtemp)
        MetaFLintegrityNonTextTestFeatures.append(FLItemp)
        MetaCDintegrityNonTextTestFeatures.append(CDItemp)
        MetaRPintegrityNonTextTestFeatures.append(RPItemp)
        MetaRGintegrityNonTextTestFeatures.append(RGItemp)
        MetaFLavailabilityNonTextTestFeatures.append(FLAtemp)
        MetaCDavailabilityNonTextTestFeatures.append(CDAtemp)
        MetaRPavailabilityNonTextTestFeatures.append(RPAtemp)
        MetaRGavailabilityNonTextTestFeatures.append(RGAtemp)

