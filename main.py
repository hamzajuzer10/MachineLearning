#!/usr/bin/python

import constants
import TrainingAndTesting
import MetaTrainingAndTesting


#run code here
if constants.meta:
    if constants.run_type == 'training':
        mconf_pred, mconf_trpred, core_directory = TrainingAndTesting.TrainAndTest()
        MetaTrainingAndTesting.MetaTrainAndTest(mconf_pred, mconf_trpred, core_directory)

    elif constants.run_type == 'classification':
        import ClassificationPredictions
else:
    if constants.run_type == 'training':
        TrainingAndTesting.TrainAndTest()
    elif constants.run_type == 'classification':
        import ClassificationPredictions
