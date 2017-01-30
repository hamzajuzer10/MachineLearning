#!/usr/bin/python

import LabelPredictions
import constants


#run code here
if constants.run_type == 'training':
    LabelPredictions.runTrainingAndTesting(constants.classifier, constants.save, constants.param_grid, constants.meta)
elif constants.run_type == 'classification':
    LabelPredictions.runClassification()
