import tensorflow as tf
from Environement.HelperFuncs.file_ops import contruct_file_path
from Environement.EnvObj import EnvObject
import os

class EnvLoader:
    def __init__(self, input_env:EnvObject):
        self.env_obj = input_env
        self.model_path = contruct_file_path(self.env_obj.model_path, "models")
        self.model_object = self.build_models()

    def build_models(self) -> dict: 
        output_dict = {}
        for model_name in os.listdir(self.model_path):
            full_path = os.path.join(self.model_path, model_name)
            output_dict[model_name] = tf.keras.models.load_model(full_path, compile=False)
        
        self.env_obj.model_object_env_init(output_dict)
        return output_dict
    
    def bring_weights(self, epoch:int) -> None:
        weight_paths = self.env_obj.training_weights_for_epoch(epoch)
        for model in weight_paths: 
            self.model_object[model].load_weights(weight_paths[model])
            self.model_object[model].compile()
        print(f"Loaded weights from epoch {epoch}")

