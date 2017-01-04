#!/usr/bin/python

#Feature extraction for the extracted Excel data
#We will have 3 separate list of features, one for C,I,A
#We will first extract the non-text features and then append the text features

import ExcelExtraction
import random
import constants


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

#Training and test features (text) - split into 2 (Description, desc, and Additional Data types, adt)
DescTrainingFeatures=[]
DescTestFeatures=[]

AdtTrainingFeatures=[]
AdtTestFeatures=[]



#Separate training and test features/ labels
#e.g. Use 95% of data as training and 5% as test

print 'separating classified training and test data...'

random.shuffle(ExcelExtraction.dict_list)
training_dict_list = ExcelExtraction.dict_list[:int(len(ExcelExtraction.dict_list)*constants.train_test_split)]
print 'No. of training data samples: ', len(training_dict_list)

test_dict_list = ExcelExtraction.dict_list[int(len(ExcelExtraction.dict_list)*constants.train_test_split):]
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
#Ensure that features are cleansed and missing values are imputed - TODO

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

    for feature in constants.FLconfidentialityNonText_features:
        FLCtemp.append(item[feature])
    for feature in constants.CDconfidentialityNonText_features:
        CDCtemp.append(item[feature])
    for feature in constants.RPconfidentialityNonText_features:
        RPCtemp.append(item[feature])
    for feature in constants.RGconfidentialityNonText_features:
        RGCtemp.append(item[feature])
    for feature in constants.FLintegrityNonText_features:
        FLItemp.append(item[feature])
    for feature in constants.CDintegrityNonText_features:
        CDItemp.append(item[feature])
    for feature in constants.RPintegrityNonText_features:
        RPItemp.append(item[feature])
    for feature in constants.RGintegrityNonText_features:
        RGItemp.append(item[feature])
    for feature in constants.FLavailabilityNonText_features:
        FLAtemp.append(item[feature])
    for feature in constants.CDavailabilityNonText_features:
        CDAtemp.append(item[feature])
    for feature in constants.RPavailabilityNonText_features:
        RPAtemp.append(item[feature])
    for feature in constants.RGavailabilityNonText_features:
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

    for feature in constants.FLconfidentialityNonText_features:
        FLCtemp.append(item[feature])
    for feature in constants.CDconfidentialityNonText_features:
        CDCtemp.append(item[feature])
    for feature in constants.RPconfidentialityNonText_features:
        RPCtemp.append(item[feature])
    for feature in constants.RGconfidentialityNonText_features:
        RGCtemp.append(item[feature])
    for feature in constants.FLintegrityNonText_features:
        FLItemp.append(item[feature])
    for feature in constants.CDintegrityNonText_features:
        CDItemp.append(item[feature])
    for feature in constants.RPintegrityNonText_features:
        RPItemp.append(item[feature])
    for feature in constants.RGintegrityNonText_features:
        RGItemp.append(item[feature])
    for feature in constants.FLavailabilityNonText_features:
        FLAtemp.append(item[feature])
    for feature in constants.CDavailabilityNonText_features:
        CDAtemp.append(item[feature])
    for feature in constants.RPavailabilityNonText_features:
        RPAtemp.append(item[feature])
    for feature in constants.RGavailabilityNonText_features:
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


#Extract training and test features (text)

print 'extracting training and text features (text)...'

for item in training_dict_list:
    DescTrainingFeatures.append(item['AppInstanceDescription'])
    AdtTrainingFeatures.append(item['AdditionalDataTypes'])


for item in test_dict_list:
    DescTestFeatures.append(item['AppInstanceDescription'])
    AdtTestFeatures.append(item['AdditionalDataTypes'])






