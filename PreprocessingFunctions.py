#!/usr/bin/python

#Prepocess labels, non-text and text features

import numpy as np
import sys
import constants

def label_transform(labelList):
    "Transform function takes a list of labels (VH,H,M,L) and transforms them into 1,2,3,4 "

    lookup = {'VH':3,'H':2,'M':1,'L':0}

    try:
        transformedList = [[lookup[labelList[i][j]] for j in range(0,len(labelList[i]))] for i in range(0,len(labelList))]
    except KeyError, e:
        print 'unexpected label values found...'
        print 'reason: "%s"' % str(e)
        print 'program is exiting...'
        sys.exit(0)

    return np.array(transformedList)

def label_inv_transform(labelList):
    "Inv transform function takes a list of transformed labels and returns their actual values VH,H,M,L"

    lookup = {3:'VH', 2:'H',1:'M',0:'L'}

    try:
        transformedList = [lookup[labelList[i]] for i in range(0,len(labelList))]
    except KeyError, e:
        print 'unexpected predicted label values found...'
        print 'reason: "%s"' % str(e)
        print 'program is exiting...'
        sys.exit(0)

    return np.array(transformedList)

def metaNonTextFeature_combine(featureList, predictedLabelList, tag):
    "Combine non-text features with predicted labels for the meta algorithm"

    lookup = {'C-FinLossPred': 0,
              'C-CusDPred': 1,
              'C-RepLossPred': 2,
              'C-RegLossPred': 3,
              'I-FinLossPred': 4,
              'I-CusDPred': 5,
              'I-RepLossPred': 6,
              'I-RegLoss': 7,
              'A-FinLossPred': 8,
              'A-CusDPred': 9,
              'A-RepLossPred': 10,
              'A-RegLossPred': 11}

    newFeatureList = []

    for sample in featureList:
        for label in constants.nMetaMetaNonText_feature[tag]:
            try:
                index = lookup[label]
                sample.append(predictedLabelList[index])
            except KeyError, e:
                print 'unexpected tags found...'
                print 'reason: "%s"' % str(e)
                print 'program is exiting...'
                sys.exit(0)

        newFeatureList.append(sample)

    return np.array(newFeatureList)

def nonTextFeature_transform(featureList):
    "Transform function takes in the list of features and converts them to integer representation"

    lookup = {"Global Banking & Markets":0,
              "Retail Banking & Wealth Management":1,
              "Payments & PCM":2,
              "Risk":3,
              "Global Finance":4,
              "Business Support":5,
              "Commercial Banking":6,
              "HOST":7,
              "HSBC Technology & Services":8,
              "Human Resources":9,
              "Global Private Banking":10,
              "North America":0,
              "Global":1,
              "EU International":2,
              "Europe":3,
              "Asia Pacific":4,
              "Middle East":5,
              "Latin America":6,
              "Tier 2":0,
              "Undefined":1,
              "Tier 1":2,
              "Tier 0":3,
              "Yes":0,
              "No":1,
              "Service partially outsourced":0,
              "Service fully outsourced":1,
              "Service managed fully from HSBC teams":2,
              "0-99k records":0,
              "100k-999k records":1,
              "1M-9M records":2,
              "10M+ records":3,
              "No customers":0,
              "<5k":1,
              "5k-499k":2,
              "500k-4.9M":3,
              "5M-49M":4,
              "50M+":5,
              "0-100":0,
              "100-1000":1,
              "1000-5,000":2,
              "5,000+":3,
              "Product Offering Details":0,
              "Transaction and Payments Details":1,
              "Customer Account":2,
              "Trade":3,
              "Contract Financial Value":4,
              "Correspondant Bank Details":5,
              "Credit Contract Details":6,
              "Customer Transaction":7,
              "Financial Instrument":8,
              "Product Contract Details":9,
              "Trade Account":10,
              "Customer Communications":11,
              "Customer Contact Details":12,
              "Customer Identification Details":13,
              "Contract Limit":14,
              "Investment Portfolio Details":15,
              "Connected Parties Identification Details":16,
              "Customer Classification":17,
              "Customer Connected Parties":18,
              "Customer Delivery Channels and Service Details":19,
              "Customer Demographic Details":20,
              "Customer External Identifier":21,
              "Software Source Code":22,
              "Investment Fund Details":23,
              "Regulatory Breach Investigation":24,
              "Confidential Project Information":25,
              "HSBC Contract Details":26,
              "Common Reporting Standard":27,
              "Connected Parties Business Activity":28,
              "IDs and Authentication":29,
              "Loan Security Details":30,
              "Customer Risk Profile":31,
              "Credit Risk Data":32,
              "Access Control Details":33,
              "Customer Complaints":34,
              "Risk and Controls Assessment Data":35,
              "Legal Advice":36,
              "Asset Valuation Data":37,
              "Counterparty Risk Data":38,
              "Economic Capital Data":39,
              "Insurance Risk Data":40,
              "Liquidity Risk Data":41,
              "Market Risk Data":42,
              "Risk Impact Data":43,
              "Risk Measure and Parameters Data":44,
              "Wholesale Credit Risk Data":45,
              "Reported Exposure Data":46,
              "Retail Pool Data":47,
              "Key Performance Metric":48,
              "FX Contract":49,
              "Insurance Policy":50,
              "Swift Keys":51,
              "Financial Measure":52,
              "Financial Statement":53,
              "Employee Demographics":54,
              "Insurance Agent Details":55,
              "Prospective Customer":56,
              "Customer Tax Filing":57,
              "Financial Crime Compliance Risk Data":58,
              "Security and Fraud Risk Data":59,
              "Legal Authority Inquiry":60,
              "Industry Mortality Rate Data":61,
              "Non-Published Regulator Views":62,
              "Building Schematics":63,
              "Contingency Site Address":64,
              "Operational Risk Data":65,
              "Traded Risk Data":66,
              "Reputational Risk Data":67,
              "Financial Stress Parameters":68,
              "Reinsurer Details":69,
              "Employee Relation Case":70,
              "Employee Biography":71,
              "Risk Strategy Data":72,
              "Growth Asset Data":73,
              "Employee Threat Evidence":74,
              "Executive Succession Planning":75,
              "Bogdan Chirila":0,
              "Anupam Kakroo":1,
              "Leo Barbaro":2,
              "Varun Kashyap":3,
              "Ben Segar":4,
              "Aakesh Pattani":5,
              "Laura Johnston":6,
              "Leon Duffield":7,
              "Jayme Metcalfe":8,
              "Warren Moore":9,
              "Hamza Juzer":10,
              "Caroline Hicks":11,
              "Raam Chandrasekharan": 12}

    try:
        transformedList = [[lookup[featureList[i][j]] for j in range(0,len(featureList[i]))] for i in range(0,len(featureList))]
    except KeyError, e:
        print 'unexpected feature values found...'
        print 'reason: "%s"' % str(e)
        print 'program is exiting...'
        sys.exit(0)

    return np.array(transformedList)
    # return transformedList

def nonTextFeature_nvalues(FeatureDict):
    "Determines the number of unique values for each feature in the dictionary"


    n_values = []

    for feature in FeatureDict:
        try:
            n_values.append(constants.nValues_feature[feature])
        except KeyError, e:
            print 'unexpected features found to determine unique number of features...'
            print 'reason: "%s"' % str(e)
            print 'program is exiting...'
            sys.exit(0)

    return n_values

def checkLabelsNFeatures(labels, features, str):
    "Function to verify correctness of labels and features"

    #TODO

    if len(labels) != len(features):
        print str,'labels and features do not have the same length...'
        print 'program is exiting...'
        sys.exit(0)