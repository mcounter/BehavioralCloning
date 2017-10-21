import os
import numpy as np
import DeepDataEngine as dd

def createStorage(
    override = False, # Override data storage
    track_v1 = False, # Include data for 1st track
    track_v2 = False, # Include data for 2nd track
    valid_set_share = 0.3): # Share of validation set
    """
    Create data storage from set of images and driving_log.csv file.
    Can combine several set of images.
    """

    if (not override) and os.path.exists('./deep_storage'):
        return

    # Generation plan - contains metadata for total data storage and it's used to generate train data set and validation data set.
    generation_plan = []

    # Image set for 1st track
    if track_v1:
        #generation_plan += dd.DeepDataEngine.createGenerationPlanFromCSV('./../../Simulator_Rec8/driving_log.csv')
        #generation_plan += dd.DeepDataEngine.createGenerationPlanFromCSV('./../../Simulator_Rec9/driving_log.csv')
        generation_plan += dd.DeepDataEngine.createGenerationPlanFromCSV('./../../Simulator_Rec14/driving_log.csv')
        generation_plan += dd.DeepDataEngine.createGenerationPlanFromCSV('./../../Simulator_Rec15/driving_log.csv')

    # Image set for 2nd track
    if track_v2:
        #generation_plan += dd.DeepDataEngine.createGenerationPlanFromCSV('./../../Simulator_Rec6/driving_log.csv')
        #generation_plan += dd.DeepDataEngine.createGenerationPlanFromCSV('./../../Simulator_Rec10/driving_log.csv')
        generation_plan += dd.DeepDataEngine.createGenerationPlanFromCSV('./../../Simulator_Rec16/driving_log.csv')
        generation_plan += dd.DeepDataEngine.createGenerationPlanFromCSV('./../../Simulator_Rec17/driving_log.csv')

    if len(generation_plan) > 0:
        # Split total generation plan on 2 sub-plans - training and validation and create data storage
        train_size = int(len(generation_plan) * (1.0 - valid_set_share))

        np.random.shuffle(generation_plan)
        plan_train = generation_plan[:train_size]
        plan_valid = generation_plan[train_size:]

        data_train = dd.DeepDataEngine('train')
        data_train.createStorage(plan_train, override = override)

        data_valid = dd.DeepDataEngine('valid')
        data_valid.createStorage(plan_valid, override = override)
