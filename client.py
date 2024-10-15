import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification
)
from accelerate import Accelerator
from peft import get_peft_model, set_peft_model_state_dict, get_peft_model_state_dict

def init_local_model(training_configs, train_dataset, loraConfig):
    dataset_name = os.path.basename(training_configs.dataset_path)
    model_name = os.path.basename(training_configs.base_model_path)
    if model_name in ["bert-base-cased"]:
        num_labels = train_dataset.features['label'].num_classes
        class_names = train_dataset.features["label"].names
        print(f"number of labels: {num_labels}")
        print(f"the labels: {class_names}")
        # Create an id2label mapping, we will need this for our classifier
        id2label = {i: label for i, label in enumerate(class_names)}
        model = AutoModelForSequenceClassification.from_pretrained(training_configs.base_model_path, id2label=id2label)
        model = get_peft_model(model, loraConfig)
    elif model_name in ["Llama-2-7b-hf", "TinyLlama-1.1B-intermediate-step-1431k-3T"]:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        device_map = {"": Accelerator().local_process_index}    # Copy the model to each device
        torch_dtype = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
            training_configs.base_model_path, 
            quantization_config=quantization_config, 
            device_map=device_map, 
            torch_dtype=torch_dtype)
        model = get_peft_model(model, loraConfig)
        model.enable_input_require_grads()

    model.print_trainable_parameters()


    return model
    
        

def set_individual_model_state_dict(client_num, lora_config, local_model, edge_model):
    target_modules = lora_config.target_modules
    local_model_dict = get_peft_model_state_dict(local_model)
    layer_dict = {}

    for layer in local_model_dict.keys():
        if layer in edge_model:
            layer_dict[layer] = edge_model[layer]

    if not layer_dict:
        raise Exception("The local model has no available target module.")
    # print(f"client{client_num}`s layer dict is")
    # print(layer_dict.keys())
    set_peft_model_state_dict(local_model, layer_dict)   
