#!/usr/bin/python

#All configurable settings and constants go here


#Machine Learning Summary Workbook
workbook_name = 'MachineLearningSummarySample.xlsm'
training_sheet_name = 'Feature Analysis - Cleansed'
actual_sheet_name = 'Actual Feature'


#Features for each CIA
FLconfidentialityNonText_features = ('AppInstLvl4BusOrg','Region','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','Classification Owners')
CDconfidentialityNonText_features = ('AppInstLvl4BusOrg','RiskFactor01','RiskFactor02','Region')
RPconfidentialityNonText_features = ('AppInstLvl4BusOrg','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','Classification Owners')
RGconfidentialityNonText_features = ('AppInstLvl4BusOrg','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','Classification Owners')
FLintegrityNonText_features = ('AppInstLvl4BusOrg','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','Classification Owners')
CDintegrityNonText_features = ('AppInstLvl4BusOrg','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','Classification Owners')
RPintegrityNonText_features = ('AppInstLvl4BusOrg','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','Classification Owners')
RGintegrityNonText_features = ('AppInstLvl4BusOrg','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','Classification Owners')
FLavailabilityNonText_features = ('AppInstLvl4BusOrg','BIA Rating','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','Classification Owners')
CDavailabilityNonText_features = ('AppInstLvl4BusOrg','BIA Rating','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','Classification Owners')
RPavailabilityNonText_features = ('AppInstLvl4BusOrg','BIA Rating','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','Classification Owners')
RGavailabilityNonText_features = ('AppInstLvl4BusOrg','BIA Rating','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','Classification Owners')

#Labels for each CIA
labels = ('C-FinLoss','C-CusD','C-RepLoss','C-RegLoss','I-FinLoss','I-CusD','I-RepLoss','I-RegLoss','A-FinLoss','A-CusD','A-RepLoss','A-RegLoss')

#Split of training and test sets
train_test_split = 0.9

#dimensionality reduction % on CIA features

#classifier used - options
#(GaussianNaiveBayes, SVM, DecisionTree, KNN)
classifier = 'DecisionTree'

# Should the training output be saved?
save = False

#classifier parameters used:
#Gaussian Bayes -


#Decision Tree -
min_samples_split = 5

