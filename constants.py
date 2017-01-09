#!/usr/bin/python

#All configurable settings and constants go here


#Machine Learning Training Workbook
training_workbook_name = 'MachineLearningTraining.xlsm'
training_sheet_name = 'Raw'

#Machine Learning Actual Workbook
actual_workbook_name = 'MachineLearningActual.xls'
actual_sheet_name = 'Raw'

#Trained classifier directory
directory='Outputs/DecisionTree_72%acc_TrainingSet_3136_2017-01-05'

#Run type ('training','classification')
run_type = 'training'

#Training Features for each CIA
#Feature descriptions
#AppInstLvl4BusOrg - Business Area
#Region - Region
#EIM Rating - BIA Impact Rating
#RiskFactor01 -
#RiskFactor02 -
#RiskFactor03 -
#RiskFactor04 -
#RiskFactor05 -
#RiskFactor06 -
#RiskFactor07 -
#RiskFactor08 -
#Category - Crown Jewel Data category
#ClassOwner - Classification Owner
#C-High - Phase 1 Confidentiality rating for Crown Jewel Data category
#I-High - Phase 1 Integrity rating for Crown Jewel Data category
#A-High - Phase 1 Availability rating for Crown Jewel Data category

#No. of unique values for each feature
nValues_feature = {'AppInstLvl4BusOrg':11,
                    'Region':7,
                    'EIM Rating':4,
                    'RiskFactor01':2,
                    'RiskFactor02':2,
                    'RiskFactor03':2,
                    'RiskFactor04':3,
                    'RiskFactor05':4,
                    'RiskFactor06':6,
                    'RiskFactor07':4,
                    'RiskFactor08':2,
                    'Category':76,
                    'ClassOwner':12,
                    'C-High':2,
                    'I-High':2,
                    'A-High':2}

#Features to be included in the training model
nNonText_feature = {'FLconfidentialityNonText_features':('AppInstLvl4BusOrg','Region','EIM Rating','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','ClassOwner','C-High',
                                     'I-High','A-High'),
                    'CDconfidentialityNonText_features':('AppInstLvl4BusOrg','Region','EIM Rating','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','ClassOwner','C-High',
                                     'I-High','A-High'),
                    'RPconfidentialityNonText_features':('AppInstLvl4BusOrg','Region','EIM Rating','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','ClassOwner','C-High',
                                     'I-High','A-High'),
                    'RGconfidentialityNonText_features':('AppInstLvl4BusOrg','Region','EIM Rating','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','ClassOwner','C-High',
                                     'I-High','A-High'),
                    'FLintegrityNonText_features':('AppInstLvl4BusOrg','Region','EIM Rating','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','ClassOwner','C-High',
                                     'I-High','A-High'),
                    'CDintegrityNonText_features':('AppInstLvl4BusOrg','Region','EIM Rating','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','ClassOwner','C-High',
                                     'I-High','A-High'),
                    'RPintegrityNonText_features':('AppInstLvl4BusOrg','Region','EIM Rating','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','ClassOwner','C-High',
                                     'I-High','A-High'),
                    'RGintegrityNonText_features':('AppInstLvl4BusOrg','Region','EIM Rating','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','ClassOwner','C-High',
                                     'I-High','A-High'),
                    'FLavailabilityNonText_features':('AppInstLvl4BusOrg','Region','EIM Rating','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','ClassOwner','C-High',
                                     'I-High','A-High'),
                    'CDavailabilityNonText_features':('AppInstLvl4BusOrg','Region','EIM Rating','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','ClassOwner','C-High',
                                     'I-High','A-High'),
                    'RPavailabilityNonText_features':('AppInstLvl4BusOrg','Region','EIM Rating','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','ClassOwner','C-High',
                                     'I-High','A-High'),
                    'RGavailabilityNonText_features':('AppInstLvl4BusOrg','Region','EIM Rating','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
                                  'RiskFactor05','RiskFactor06','RiskFactor07','RiskFactor08','Category','ClassOwner','C-High',
                                     'I-High','A-High')}


#Labels for each CIA
labels = ('C-FinLoss','C-CusD','C-RepLoss','C-RegLoss','I-FinLoss','I-CusD','I-RepLoss','I-RegLoss','A-FinLoss','A-CusD','A-RepLoss','A-RegLoss')

#Split of training and test sets (0<value<=1)
train_test_split = 1.0

#dimensionality reduction % on CIA features

#classifier used - options
#(GaussianNaiveBayes, SVM, DecisionTree, KNN, RandomForest, AdaBoostDT)
classifier = {'FLconfidentiality':'RandomForest',
              'CDconfidentiality':'AdaBoostDT',
              'RPconfidentiality':'AdaBoostDT',
              'RGconfidentiality':'AdaBoostDT',
              'FLintegrity':'AdaBoostDT',
              'CDintegrity':'AdaBoostDT',
              'RPintegrity':'RandomForest',
              'RGintegrity':'AdaBoostDT',
              'FLavailability':'RandomForest',
              'CDavailability':'AdaBoostDT',
              'RPavailability':'RandomForest',
              'RGavailability':'RandomForest'}


# Should the training output be saved?
save = True

#classifier parameters used
#Examples -
#Gaussian Bayes:
#None

#Decision Tree:
# param_grid = [
#   {'min_samples_split': [2, 5, 10, 20], 'criterion': ['gini']},
#   {'min_samples_split': [2, 5, 10, 20], 'criterion': ['entropy']},
#  ]

#RandomForest
#param_grid = {'n_estimators': [5, 10, 15], 'min_samples_split': [2, 5, 10, 20], 'min_samples_leaf': [1,5,10], 'n_jobs':[-1]}

#SVM
# param_grid = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
#  ]

#Nearest Neighbors
# param_grid = {'n_neighbors': [5, 10, 15], 'weights': ['uniform','distance'], 'p':[1,2], 'n_jobs':[-1]}

#Adaboost with DecisionTrees
#param_grid = {'base_estimator__criterion' : ['gini'], 'base_estimator__min_samples_split' :[2, 5, 10, 20], 'n_estimators': [30, 40, 50]}

param_grid = {'FLconfidentiality':{'n_estimators': [15,20], 'min_samples_split': [2], 'min_samples_leaf': [1], 'n_jobs':[-1]},
              'CDconfidentiality':{'base_estimator__criterion' : ['gini'], 'base_estimator__min_samples_split' :[5], 'n_estimators': [50, 60]},
              'RPconfidentiality':{'base_estimator__criterion' : ['gini'], 'base_estimator__min_samples_split' :[5], 'n_estimators': [50, 60]},
              'RGconfidentiality':{'base_estimator__criterion' : ['gini'], 'base_estimator__min_samples_split' :[5], 'n_estimators': [35, 40, 45]},
              'FLintegrity':{'base_estimator__criterion' : ['gini'], 'base_estimator__min_samples_split' :[5], 'n_estimators': [50, 60]},
              'CDintegrity':{'base_estimator__criterion' : ['gini'], 'base_estimator__min_samples_split' :[5], 'n_estimators': [50, 60]},
              'RPintegrity':{'n_estimators': [15,20], 'min_samples_split': [2], 'min_samples_leaf': [1], 'n_jobs':[-1]},
              'RGintegrity':{'base_estimator__criterion' : ['gini'], 'base_estimator__min_samples_split' :[5], 'n_estimators': [35, 40, 45]},
              'FLavailability':{'n_estimators': [15,20], 'min_samples_split': [2], 'min_samples_leaf': [1], 'n_jobs':[-1]},
              'CDavailability':{'base_estimator__criterion' : ['gini'], 'base_estimator__min_samples_split' :[2], 'n_estimators': [50, 60]},
              'RPavailability':{'n_estimators': [15,20], 'min_samples_split': [2], 'min_samples_leaf': [1], 'n_jobs':[-1]},
              'RGavailability':{'n_estimators': [15,20], 'min_samples_split': [2], 'min_samples_leaf': [1], 'n_jobs':[-1]}}