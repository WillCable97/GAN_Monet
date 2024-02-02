##################PACKAGESS##################
#Public imports
import os
import tensorflow as tf
import json
import matplotlib.pyplot as plt

#Base imports
from Environement.EnvObj import EnvObject

from Environement.HelperFuncs.file_ops import read_json_file

ModelName = 'GanV3'

##################ENVIRONMENT####################
parent_dir  = os.path.abspath('.')
vis_folder = os.path.join(parent_dir, 'src', 'visualization', 'EpochCompare')
env_object = EnvObject(os.path.abspath('.'), ModelName)
latest_epoch = env_object.read_epoch_number()

json_obj = read_json_file(env_object.error_file)


#print(json_obj)
for key in json_obj['1']: print(key) #epoch -> model name ->


#Epoch Averaged

averaged_model_errors = {}


for epoch_key in json_obj: 
    for k, model_name in enumerate(json_obj[epoch_key]):
        if not model_name in averaged_model_errors: averaged_model_errors[model_name] = []
        model_list_for_epoch = [sublist[k] for sublist in json_obj[epoch_key][model_name]]
        average_error_for_epoch = sum(model_list_for_epoch) / float(len(model_list_for_epoch))
        averaged_model_errors[model_name].append(average_error_for_epoch)

print(averaged_model_errors)



fig, axs = plt.subplots(2, 2, figsize=(10, 8))

for k, model_name in enumerate(averaged_model_errors):
    column = k % 2
    row = int(k / 2)
    axs[row,column].plot(averaged_model_errors[model_name])
    axs[row,column].set_title(model_name)

plt.tight_layout()
plt.show()












