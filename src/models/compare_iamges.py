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
vis_folder = os.path.join(parent_dir, 'src', 'visualization', 'EpochCompare')
final_v_og_folder = os.path.join(vis_folder, 'FinalVOg')
epoch_progress_folder = os.path.join(vis_folder, 'EpochProgression')
env_object = EnvObject(os.path.abspath('.'), ModelName)
env_loader = EnvLoader(env_object)
latest_epoch = env_object.read_epoch_number()

##################FEATURE MAPS##################
dat_col_name = 'text_feature' #Update
feature_description = {dat_col_name: tf.io.FixedLenFeature([], dtype=tf.string),}

##################DATA##################
processed_data_folder = os.path.join(parent_dir, 'data', 'processed')
training_list = ['trainB'] #Update
records_paths = [os.path.join(os.path.join(processed_data_folder, f_name)) for f_name in training_list]
domain_containers = [DataContainer.TensorflowDataObject(records_path, feature_description, dat_col_name) for records_path in records_paths]
pipline_proc = ProcessorUsed()


#Original vs final compare
env_loader.bring_weights(latest_epoch)
for i in range(30):
    #Original Image
    raw_example = domain_containers[0].return_single_eg()
    proc_example = pipline_proc.preprocess_function(raw_example)
    original_image = pipline_proc.postprocess_function(proc_example)

    #Run through the final model
    raw_generated = env_loader.model_object['b_a_generator'](proc_example)
    output_generated = pipline_proc.postprocess_function(raw_generated)

    plt.imsave(os.path.join(final_v_og_folder, f"{i}_original.jpg"),original_image)
    plt.imsave(os.path.join(final_v_og_folder, f"{i}_final.jpg"),output_generated)

"""
#Original vs final compare (Reversed)
env_loader.bring_weights(latest_epoch)
for i in range(5):
    #Original Image
    raw_example = domain_containers[0].return_single_eg()
    proc_example = pipline_proc.preprocess_function(raw_example)
    original_image = pipline_proc.postprocess_function(proc_example)

    #Run through the final model
    raw_generated = env_loader.model_object['a_b_generator'](proc_example)
    output_generated = pipline_proc.postprocess_function(raw_generated)

    plt.imsave(os.path.join(final_v_og_folder, f"{i+5}_original.jpg"),original_image)
    plt.imsave(os.path.join(final_v_og_folder, f"{i+5}_final.jpg"),output_generated)
"""





"""
#Progressions
scaling_factor = latest_epoch/ 100 

raw_example = domain_containers[0].return_single_eg()
proc_example = pipline_proc.preprocess_function(raw_example)
original_image = pipline_proc.postprocess_function(proc_example)


for k, i in enumerate([25,50,100]): #Need to add in 75th quantile when weights are loaded 
    epoch = int(i * scaling_factor)
    env_loader.bring_weights(int(i * scaling_factor))
    raw_generated = env_loader.model_object['b_a_generator'](proc_example)
    output_generated = pipline_proc.postprocess_function(raw_generated)

    plt.imsave(os.path.join(epoch_progress_folder, f"1_Epoch_{epoch}.jpg"),output_generated)


plt.imsave(os.path.join(epoch_progress_folder, "1_original.jpg"),original_image)
"""




