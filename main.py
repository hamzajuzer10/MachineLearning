#!/usr/bin/python

import constants


#run code here
if constants.meta:
    if constants.run_type == 'training':
        import TrainingAndTesting
    elif constants.run_type == 'classification':
        import ClassificationPredictions
else:
    if constants.run_type == 'training':
        import MetaTrainingAndTesting
    elif constants.run_type == 'classification':
        import ClassificationPredictions
