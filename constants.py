#!/usr/bin/python

#All configurable settings and constants go here


#Machine Learning Training Workbook
training_workbook_name = 'MachineLearningTraining.xlsm'
training_sheet_name = 'Raw'

#Machine Learning Actual Workbook
actual_workbook_name = 'Classification/MachineLearningActual_2706groups.xls'
actual_sheet_name = 'Raw'

#Trained classifier directory
directory='Outputs/TrainingSet_7439_2017-01-13_17h45m'

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
                    'ClassOwner':14,
                    'C-High':2,
                    'I-High':2,
                    'A-High':2,
                    'C-FinLossPred':4,
                    'C-CusDPred':4,
                    'C-RepLossPred':4,
                    'C-RegLossPred':4,
                    'I-FinLossPred':4,
                    'I-CusDPred':4,
                    'I-RepLossPred':4,
                    'I-RegLoss':4,
                    'A-FinLossPred':4,
                    'A-CusDPred':4,
                    'A-RepLossPred':4,
                    'A-RegLossPred':4}


#Features to be included in the L1 training model
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



#L1 classifier used
#Options are: GaussianNaiveBayes, SVM, DecisionTree, KNN, RandomForest, AdaBoostDT
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

#L1 classifier parameters used
#Options are:
#Gaussian Bayes -

#Decision Tree -
# param_grid = [
#   {'min_samples_split': [2, 5, 10, 20], 'criterion': ['gini']},
#   {'min_samples_split': [2, 5, 10, 20], 'criterion': ['entropy']},
#  ]

#RandomForest -
#param_grid = {'n_estimators': [5, 10, 15], 'min_samples_split': [2, 5, 10, 20], 'min_samples_leaf': [1,5,10], 'n_jobs':[-1]}

#SVM
# param_grid = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
#  ]

#Nearest Neighbors -
# param_grid = {'n_neighbors': [5, 10, 15], 'weights': ['uniform','distance'], 'p':[1,2], 'n_jobs':[-1]}

#Adaboost with DecisionTrees -
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



#L2 meta classifier
meta = True



#Features to be included in the L2 (meta) training model
nMetaBaseNonText_feature = {'FLconfidentialityNonText_features':('AppInstLvl4BusOrg','Region','EIM Rating','RiskFactor01','RiskFactor02','RiskFactor03','RiskFactor04',
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

nMetaMetaNonText_feature = {'FLconfidentialityNonText_features':('C-FinLossPred','C-CusDPred','C-RepLossPred','C-RegLossPred','I-FinLossPred','I-CusDPred','I-RepLossPred','I-RegLoss',
                                     'A-FinLossPred','A-CusDPred','A-RepLossPred','A-RegLossPred'),
                    'CDconfidentialityNonText_features':('C-FinLossPred','C-CusDPred','C-RepLossPred','C-RegLossPred','I-FinLossPred','I-CusDPred','I-RepLossPred','I-RegLoss',
                                     'A-FinLossPred','A-CusDPred','A-RepLossPred','A-RegLossPred'),
                    'RPconfidentialityNonText_features':('C-FinLossPred','C-CusDPred','C-RepLossPred','C-RegLossPred','I-FinLossPred','I-CusDPred','I-RepLossPred','I-RegLoss',
                                     'A-FinLossPred','A-CusDPred','A-RepLossPred','A-RegLossPred'),
                    'RGconfidentialityNonText_features':('C-FinLossPred','C-CusDPred','C-RepLossPred','C-RegLossPred','I-FinLossPred','I-CusDPred','I-RepLossPred','I-RegLoss',
                                     'A-FinLossPred','A-CusDPred','A-RepLossPred','A-RegLossPred'),
                    'FLintegrityNonText_features':('C-FinLossPred','C-CusDPred','C-RepLossPred','C-RegLossPred','I-FinLossPred','I-CusDPred','I-RepLossPred','I-RegLoss',
                                     'A-FinLossPred','A-CusDPred','A-RepLossPred','A-RegLossPred'),
                    'CDintegrityNonText_features':('C-FinLossPred','C-CusDPred','C-RepLossPred','C-RegLossPred','I-FinLossPred','I-CusDPred','I-RepLossPred','I-RegLoss',
                                     'A-FinLossPred','A-CusDPred','A-RepLossPred','A-RegLossPred'),
                    'RPintegrityNonText_features':('C-FinLossPred','C-CusDPred','C-RepLossPred','C-RegLossPred','I-FinLossPred','I-CusDPred','I-RepLossPred','I-RegLoss',
                                     'A-FinLossPred','A-CusDPred','A-RepLossPred','A-RegLossPred'),
                    'RGintegrityNonText_features':('C-FinLossPred','C-CusDPred','C-RepLossPred','C-RegLossPred','I-FinLossPred','I-CusDPred','I-RepLossPred','I-RegLoss',
                                     'A-FinLossPred','A-CusDPred','A-RepLossPred','A-RegLossPred'),
                    'FLavailabilityNonText_features':('C-FinLossPred','C-CusDPred','C-RepLossPred','C-RegLossPred','I-FinLossPred','I-CusDPred','I-RepLossPred','I-RegLoss',
                                     'A-FinLossPred','A-CusDPred','A-RepLossPred','A-RegLossPred'),
                    'CDavailabilityNonText_features':('C-FinLossPred','C-CusDPred','C-RepLossPred','C-RegLossPred','I-FinLossPred','I-CusDPred','I-RepLossPred','I-RegLoss',
                                     'A-FinLossPred','A-CusDPred','A-RepLossPred','A-RegLossPred'),
                    'RPavailabilityNonText_features':('C-FinLossPred','C-CusDPred','C-RepLossPred','C-RegLossPred','I-FinLossPred','I-CusDPred','I-RepLossPred','I-RegLoss',
                                     'A-FinLossPred','A-CusDPred','A-RepLossPred','A-RegLossPred'),
                    'RGavailabilityNonText_features':('C-FinLossPred','C-CusDPred','C-RepLossPred','C-RegLossPred','I-FinLossPred','I-CusDPred','I-RepLossPred','I-RegLoss',
                                     'A-FinLossPred','A-CusDPred','A-RepLossPred','A-RegLossPred')}

#L1 classifier used
#Options are: GaussianNaiveBayes, SVM, DecisionTree, KNN, RandomForest, AdaBoostDT
meta_classifier = {'FLconfidentiality':'RandomForest',
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

#L2 classifier parameters used
#Options are:
#Gaussian Bayes -

#Decision Tree -
# param_grid = [
#   {'min_samples_split': [2, 5, 10, 20], 'criterion': ['gini']},
#   {'min_samples_split': [2, 5, 10, 20], 'criterion': ['entropy']},
#  ]

#RandomForest -
#param_grid = {'n_estimators': [5, 10, 15], 'min_samples_split': [2, 5, 10, 20], 'min_samples_leaf': [1,5,10], 'n_jobs':[-1]}

#SVM
# param_grid = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
#  ]

#Nearest Neighbors -
# param_grid = {'n_neighbors': [5, 10, 15], 'weights': ['uniform','distance'], 'p':[1,2], 'n_jobs':[-1]}

#Adaboost with DecisionTrees -
#param_grid = {'base_estimator__criterion' : ['gini'], 'base_estimator__min_samples_split' :[2, 5, 10, 20], 'n_estimators': [30, 40, 50]}

meta_param_grid = {'FLconfidentiality':{'n_estimators': [15,20], 'min_samples_split': [2], 'min_samples_leaf': [1], 'n_jobs':[-1]},
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


#Labels for each CIA
labels = ('C-FinLoss','C-CusD','C-RepLoss','C-RegLoss','I-FinLoss','I-CusD','I-RepLoss','I-RegLoss','A-FinLoss','A-CusD','A-RepLoss','A-RegLoss')

#Split of training and test sets (0<value<=1)
train_test_split = 0.8

# Should the training output be saved?
save = True

