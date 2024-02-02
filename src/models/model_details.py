##################PACKAGESS##################
#Public imports
import os
import tensorflow as tf
import matplotlib.pyplot as plt

#Local imports
from  Processors.GanImageProcessor import GanImageProcessor as ProcessorUsed #Change to base class again

#Base imports
import DataHandling.DataContainer as DataContainer
from Environement.EnvLoader import EnvLoader
from Environement.EnvObj import EnvObject

ModelName = 'GanV3'

##################ENVIRONMENT####################
parent_dir  = os.path.abspath('.')
env_object = EnvObject(os.path.abspath('.'), ModelName)
env_loader = EnvLoader(env_object)
latest_epoch = env_object.read_epoch_number()
env_loader.bring_weights(latest_epoch)
env_loader.model_object['b_a_generator'].summary()




