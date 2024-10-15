import copy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
import numpy as np
from peft import get_peft_model_state_dict,set_peft_model_state_dict
from server import init_global_model, edge_aggregate_layer,split_layers, global_aggregate_matrix
from config import get_config,save_config,get_training_args, get_trainer
from dataset import preprocess_dateset,split_dataset
from client import init_local_model,set_individual_model_state_dict


# ===== get training config =====
training_configs = get_config()
save_config(training_configs)
model_name = os.path.basename(training_configs.base_model_path)

k_group_clients = training_configs.k_group_clients
q_group_clients = training_configs.q_group_clients
v_group_clients = training_configs.v_group_clients
# for llama and tiny llama
o_group_clients = training_configs.o_group_clients



# ===== Load and splite the dataset =====
train_dataset, eval_dataset = preprocess_dateset()
local_train_datasets,local_eval_datasets = split_dataset(train_dataset, eval_dataset)
sample_num_list = [len(local_train_datasets[i]) for i in range(training_configs.num_clients)]       # for fedeavg aggregation
print(f"sample_num_list:{sample_num_list}") 

# ===== init FL global model and distribute to clients =====
model, global_dict = init_global_model(training_configs, train_dataset)
local_dict_list = [copy.deepcopy(global_dict) for i in range(training_configs.num_clients)]



# ===== Start federated training =====
k_group_training_loss = [[] for i in range(len(k_group_clients))]
q_group_training_loss = [[] for i in range(len(q_group_clients))]
v_group_training_loss = [[] for i in range(len(v_group_clients))]
# for llama and tiny llama
o_group_training_loss = [[] for i in range(len(o_group_clients))]

for global_round in range(training_configs.num_rounds):        # main loop of LLM FL training
    if model_name in ["bert-base-cased"]:
        clients_this_round = k_group_clients+q_group_clients+v_group_clients   # client id start from 0
        k_lora, q_lora, v_lora = split_layers(global_dict,model_name)  # Before setting state dict, you need to process global dict into separate dicts of k, q and v
    elif model_name in ["Llama-2-7b-hf", "TinyLlama-1.1B-intermediate-step-1431k-3T"]:
        clients_this_round = k_group_clients+q_group_clients+v_group_clients+o_group_clients    # client id start from 0
        k_lora, q_lora, v_lora, o_lora = split_layers(global_dict,model_name)  # Before setting state dict, you need to process global dict into separate dicts of k, q, v and o
    print(f">> ==================== Global round {global_round+1} : {clients_this_round} ====================")
    # edge server aggregating round
    for edge_round in range(training_configs.edge_aggregate_round):
        print(f">> ==================== Edge round {edge_round+1} ====================")
        # ===== Start o group training =====  
        if model_name in ["Llama-2-7b-hf", "TinyLlama-1.1B-intermediate-step-1431k-3T"]:   
            for client in o_group_clients:
                print("start o group training")
                local_model = init_local_model(training_configs, train_dataset, get_config(client))     # every client has its own local model
                set_individual_model_state_dict(client, get_config(client),local_model,o_lora)
                training_args = get_training_args(training_configs)
                trainer = get_trainer(training_configs,local_model,training_args,local_train_datasets[client],local_eval_datasets[client])
                results = trainer.train()
                o_group_training_loss[client-len(k_group_clients)-len(q_group_clients)-len(v_group_clients)].append(results.training_loss)

                # ===== Save the clients` local model =====
                # lora_model_path = os.path.join(training_configs.output_dir, f"checkpoint-global-{global_round+1}-edge-{edge_round+1}-{client}")
                # trainer.save_model(lora_model_path)
                local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(local_model))   # copy is needed!
            o_edge_dict = edge_aggregate_layer(local_dict_list)
            set_peft_model_state_dict(model, o_edge_dict)   # Update q edge model

            # Save the last edge model
            if edge_round == training_configs.edge_aggregate_round-1:
                trainer.save_model(os.path.join(training_configs.output_dir, f"checkpoint-global-{global_round+1}-edge-o"))
                np.save(os.path.join(training_configs.output_dir, "o_group_training_loss.npy"), np.array(o_group_training_loss))

        # ===== Start k group training =====
        for client in k_group_clients:
            print("start k group training")
            local_model = init_local_model(training_configs, train_dataset, get_config(client))     # every client has its own local model
            set_individual_model_state_dict(client, get_config(client),local_model,k_lora)
            training_args = get_training_args(training_configs)
            trainer = get_trainer(training_configs,local_model,training_args,local_train_datasets[client],local_eval_datasets[client])
            results = trainer.train()
            k_group_training_loss[client].append(results.training_loss)

            # ===== Save the clients` local model =====
            # lora_model_path = os.path.join(training_configs.output_dir, f"checkpoint-global-{global_round+1}-edge-{edge_round+1}-{client}")
            # trainer.save_model(lora_model_path)
            local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(local_model))   # copy is needed!
        k_edge_dict = edge_aggregate_layer(local_dict_list)
        set_peft_model_state_dict(model, k_edge_dict)   # Update k edge model
        # Save the last edge model
        if edge_round == training_configs.edge_aggregate_round-1:
            trainer.save_model(os.path.join(training_configs.output_dir, f"checkpoint-global-{global_round+1}-edge-k"))
            np.save(os.path.join(training_configs.output_dir, "k_group_training_loss.npy"), np.array(k_group_training_loss))


        # ===== Start q group training =====
        for client in q_group_clients:
            print("start q group training")
            local_model = init_local_model(training_configs, train_dataset, get_config(client))     # every client has its own local model
            set_individual_model_state_dict(client, get_config(client),local_model,q_lora)
            training_args = get_training_args(training_configs)
            trainer = get_trainer(training_configs,local_model,training_args,local_train_datasets[client],local_eval_datasets[client])
            results = trainer.train()
            q_group_training_loss[client-len(k_group_clients)].append(results.training_loss)

            # ===== Save the clients` local model =====
            # lora_model_path = os.path.join(training_configs.output_dir, f"checkpoint-global-{global_round+1}-edge-{edge_round+1}-{client}")
            # trainer.save_model(lora_model_path)
            local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(local_model))   # copy is needed!
        q_edge_dict = edge_aggregate_layer(local_dict_list)
        set_peft_model_state_dict(model, q_edge_dict)   # Update q edge model

        # Save the last edge model
        if edge_round == training_configs.edge_aggregate_round-1:
            trainer.save_model(os.path.join(training_configs.output_dir, f"checkpoint-global-{global_round+1}-edge-q"))
            np.save(os.path.join(training_configs.output_dir, "q_group_training_loss.npy"), np.array(q_group_training_loss))
        
        # ===== Start q group training =====
        for client in v_group_clients:
            print("start v group training")
            local_model = init_local_model(training_configs, train_dataset, get_config(client))     # every client has its own local model
            set_individual_model_state_dict(client, get_config(client),local_model,v_lora)
            training_args = get_training_args(training_configs)
            trainer = get_trainer(training_configs,local_model,training_args,local_train_datasets[client],local_eval_datasets[client])
            results = trainer.train()
            v_group_training_loss[client-len(k_group_clients)-len(q_group_clients)].append(results.training_loss)

            # ===== Save the clients` local model =====
            # lora_model_path = os.path.join(training_configs.output_dir, f"checkpoint-global-{global_round+1}-edge-{edge_round+1}-{client}")
            # trainer.save_model(lora_model_path)
            local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(local_model))   # copy is needed!
        v_edge_dict = edge_aggregate_layer(local_dict_list)
        set_peft_model_state_dict(model, v_edge_dict)   # Update q edge model

        # Save the last edge model
        if edge_round == training_configs.edge_aggregate_round-1:
            trainer.save_model(os.path.join(training_configs.output_dir, f"checkpoint-global-{global_round+1}-edge-v"))
            np.save(os.path.join(training_configs.output_dir, "v_group_training_loss.npy"), np.array(v_group_training_loss))
        


    # ===== Server aggregates the local models =====
    if model_name in ["bert-base-cased"]:
        global_dict = global_aggregate_matrix(k_edge_dict, q_edge_dict, v_edge_dict)
    elif model_name in ["Llama-2-7b-hf", "TinyLlama-1.1B-intermediate-step-1431k-3T"]:
        global_dict = global_aggregate_matrix(k_edge_dict, q_edge_dict, v_edge_dict, o_edge_dict)
        
    set_peft_model_state_dict(model, global_dict)   # Update global model


    # ===== Save the global model =====
    trainer.save_model(os.path.join(training_configs.output_dir, f"checkpoint-{global_round+1}"))