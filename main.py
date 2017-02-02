#!/usr/bin/python

import constants
import TrainingAndTesting


#run code here
if constants.meta:
    if constants.run_type == 'training':
        mconf_pred, mconf_trpred = TrainingAndTesting.TrainAndTest()

    elif constants.run_type == 'classification':
        import ClassificationPredictions
else:
    if constants.run_type == 'training':
        TrainAndTest()
    elif constants.run_type == 'classification':
        import ClassificationPredictions
