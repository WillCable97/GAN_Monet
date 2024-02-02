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
from keras.losses import BinaryCrossentropy

ModelName = 'GanV3'

##################ENVIRONMENT####################
parent_dir  = os.path.abspath('.')
vis_folder = os.path.join(parent_dir, 'src', 'visualization', 'EpochCompare')
final_v_og_folder = os.path.join(vis_folder, 'FinalVOg')
epoch_progress_folder = os.path.join(vis_folder, 'EpochProgression')
env_object = EnvObject(os.path.abspath('.'), ModelName)
env_loader = EnvLoader(env_object)
gen_loader = EnvLoader(env_object)
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




entropy_loss = BinaryCrossentropy(from_logits=True) 

raw_image_list = []

for i in range(50):
    raw_image_list.append(domain_containers[0].return_single_eg())

#print(raw_image_list)
#proc_example = pipline_proc.preprocess_function(raw_example)


#Read in first epoch generator
#env_loader.bring_weights(1)
#first_gen_generator = env_loader.model_object['b_a_generator']


heat_list = []
heat_list_d = []
steps_taken = 10

for j in range(1,latest_epoch,steps_taken):
    if j ==0 :continue
    #if j >=10: continue 
    gen_loader.bring_weights(int(j))
    first_gen_generator = gen_loader.model_object['b_a_generator']
    loss_list = []
    loss_list_d = []

    for i in range(1,latest_epoch,steps_taken):
        if i ==0 :continue
        #if i >=10: continue 

        env_loader.bring_weights(int(i))
        epoch_descrim = env_loader.model_object['a_descrim']

        losses_for_epoch = []
        losses_for_epoch_d = []

        for image in raw_image_list:
            proc_image = pipline_proc.preprocess_function(image)
            raw_generated = first_gen_generator(proc_image)
            discrim_result = epoch_descrim(raw_generated)
            loss_val = entropy_loss(tf.ones_like(discrim_result), discrim_result).numpy()
            losses_for_epoch.append(loss_val)

            loss_val_d = entropy_loss(tf.zeros_like(discrim_result), discrim_result).numpy()
            losses_for_epoch_d.append(loss_val_d)


        average_for_epoch = sum(losses_for_epoch) / float(len(losses_for_epoch))
        average_for_epoch_d = sum(losses_for_epoch_d) / float(len(losses_for_epoch_d))
        loss_list.append(average_for_epoch)
        loss_list_d.append(average_for_epoch_d)

    heat_list.append(loss_list)
    heat_list_d.append(loss_list_d)







plt.pcolormesh(range(1,latest_epoch,steps_taken), range(1,latest_epoch,steps_taken),heat_list_d, cmap='inferno')
plt.colorbar()
plt.xlabel("Discriminator Epoch")
plt.ylabel("Generator Epoch")
plt.title("Discriminator Loss")
plt.savefig("v1_disloss")


plt.clf()


plt.pcolormesh(range(1,latest_epoch,steps_taken), range(1,latest_epoch,steps_taken),heat_list, cmap='inferno')
plt.colorbar()
plt.xlabel("Discriminator Epoch")
plt.ylabel("Generator Epoch")
plt.title("Generator Loss")
plt.savefig("v1_genloss")





"""
for i in range(latest_epoch):
    if i ==0 :continue

    env_loader.bring_weights(int(i))

    #first_gen_generator = env_loader.model_object['b_a_generator']
    epoch_descrim = env_loader.model_object['a_descrim']

    losses_for_epoch = []



    for image in raw_image_list:
        proc_image = pipline_proc.preprocess_function(image)
        raw_generated = first_gen_generator(proc_image)
        discrim_result = epoch_descrim(raw_generated)
        loss_val = entropy_loss(tf.zeros_like(discrim_result), discrim_result).numpy()
        losses_for_epoch.append(loss_val)


    average_for_epoch = sum(losses_for_epoch) / float(len(losses_for_epoch))
    loss_list.append(average_for_epoch)

print(loss_list)


import matplotlib.pyplot as plt

plt.plot(loss_list)
plt.show()
"""


"""
#Original vs final compare
env_loader.bring_weights(latest_epoch)
for i in range(5):
    #Original Image
    raw_example = domain_containers[0].return_single_eg()
    proc_example = pipline_proc.preprocess_function(raw_example)
    original_image = pipline_proc.postprocess_function(proc_example)

    #Run through the final model
    raw_generated = env_loader.model_object['b_a_generator'](proc_example)
    output_generated = pipline_proc.postprocess_function(raw_generated)

    plt.imsave(os.path.join(final_v_og_folder, f"{i}_original.jpg"),original_image)
    plt.imsave(os.path.join(final_v_og_folder, f"{i}_final.jpg"),output_generated)





raw_generated = env_loader.model_object['b_a_generator'](proc_example)




#raw_generated = env_loader.model_object['b_a_generator'](proc_example)

"""


























"""
#Original vs final compare
env_loader.bring_weights(latest_epoch)
for i in range(5):
    #Original Image
    raw_example = domain_containers[0].return_single_eg()
    proc_example = pipline_proc.preprocess_function(raw_example)
    original_image = pipline_proc.postprocess_function(proc_example)

    #Run through the final model
    raw_generated = env_loader.model_object['b_a_generator'](proc_example)
    output_generated = pipline_proc.postprocess_function(raw_generated)

    plt.imsave(os.path.join(final_v_og_folder, f"{i}_original.jpg"),original_image)
    plt.imsave(os.path.join(final_v_og_folder, f"{i}_final.jpg"),output_generated)




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



