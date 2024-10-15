import os
import json
from peft import  LoraConfig
from dataclasses import dataclass, field, asdict
from typing import Optional,List
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
)
from trl import DataCollatorForCompletionOnlyLM,SFTTrainer
from datetime import datetime, timedelta



LConfigs = []       # Create the individual config of clients

# the edge k server and k group for bert
# the edge k server and k group for tiny llama

LConfig0 = LoraConfig(
    # lora config for bert        
        # r=8,
        # lora_alpha=32,
        # target_modules=["key"],
        # layers_to_transform=[0, 1, 2, 3, 8, 9, 10, 11],
        # lora_dropout=0.05,
        # bias="none",
        # task_type="SEQ_CLS"
    # lora config for tiny llama
        task_type='CAUSAL_LM',
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['k_proj'],
        layers_to_transform=[0, 1, 2, 3, 4, 17, 18, 19, 20, 21],
        # layers_to_transform=[0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31],
        bias='none',
    )

LConfigs.append(LConfig0)

LConfig1 = LoraConfig(
    # lora config for bert  
        # r=8,
        # lora_alpha=32,
        # target_modules=["key"],
        # layers_to_transform=[0, 1, 2, 3],
        # lora_dropout=0.05,
        # bias="none",
        # task_type="SEQ_CLS"
    # lora config for tiny llama
        task_type='CAUSAL_LM',
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['k_proj'],
        layers_to_transform=[0, 1, 2, 3, 4],
        # layers_to_transform=[0, 1, 2, 3, 4, 5, 6, 7],
        bias='none',
    )
LConfigs.append(LConfig1)

LConfig2 = LoraConfig(
    # lora config for bert 
        # r=8,
        # lora_alpha=32,
        # target_modules=["key"],
        # layers_to_transform=[4, 5, 6, 7],
        # lora_dropout=0.05,
        # bias="none",
        # task_type="SEQ_CLS"
    # lora config for tiny llama
        task_type='CAUSAL_LM',
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['k_proj'],
        layers_to_transform=[17, 18, 19, 20, 21],
        # layers_to_transform=[24, 25, 26, 27, 28, 29, 30, 31],
        bias='none',
    )
LConfigs.append(LConfig2)

LConfig3 = LoraConfig(
    # lora config for bert  
        # r=8,
        # lora_alpha=32,
        # target_modules=["key"],
        # layers_to_transform=[8, 9, 10, 11],
        # lora_dropout=0.05,
        # bias="none",
        # task_type="SEQ_CLS"
    # lora config for tiny llama
        task_type='CAUSAL_LM',
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        # target_modules=['k_proj'],
        target_modules=['q_proj'],
        layers_to_transform=[0, 1, 2, 3, 4, 17, 18, 19, 20, 21],
        # layers_to_transform=[0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31],
        bias='none',
    )
LConfigs.append(LConfig3)

# the edge q server and q group for bert 
# the edge q server and q group for tiny llama
LConfig4 = LoraConfig(
    # lora config for bert  
        # r=8,
        # lora_alpha=32,
        # target_modules=["query"],
        # layers_to_transform=[0, 1, 2, 3, 8, 9, 10, 11],
        # lora_dropout=0.05,
        # bias="none",
        # task_type="SEQ_CLS"
    # lora config for tiny llama
        task_type='CAUSAL_LM',
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q_proj'],
        layers_to_transform=[0, 1, 2, 3, 4],
        # layers_to_transform=[0, 1, 2, 3, 4, 5, 6, 7],
        bias='none',
    )
LConfigs.append(LConfig4)

LConfig5 = LoraConfig(
    # lora config for bert  
        # r=8,
        # lora_alpha=32,
        # target_modules=["query"],
        # layers_to_transform=[0, 1, 2, 3],
        # lora_dropout=0.05,
        # bias="none",
        # task_type="SEQ_CLS"
    # lora config for tiny llama
        task_type='CAUSAL_LM',
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q_proj'],
        layers_to_transform=[17, 18, 19, 20, 21],
        # layers_to_transform=[24, 25, 26, 27, 28, 29, 30, 31],
        bias='none',
    )
LConfigs.append(LConfig5)


LConfig6 = LoraConfig(
    # lora config for bert  
        # r=8,
        # lora_alpha=32,
        # target_modules=["query"],
        # layers_to_transform=[4, 5, 6, 7],
        # lora_dropout=0.05,
        # bias="none",
        # task_type="SEQ_CLS"
    # lora config for tiny llama
        task_type='CAUSAL_LM',
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        # target_modules=['q_proj'],
        target_modules=['v_proj'],
        layers_to_transform=[0, 1, 2, 3, 4, 17, 18, 19, 20, 21],
        # layers_to_transform=[0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31],
        bias='none',
    )
LConfigs.append(LConfig6)

LConfig7 = LoraConfig(
    # lora config for bert  
        # r=8,
        # lora_alpha=32,
        # target_modules=["query"],
        # layers_to_transform=[8, 9, 10, 11],
        # lora_dropout=0.05,
        # bias="none",
        # task_type="SEQ_CLS"
    # lora config for tiny llama
        task_type='CAUSAL_LM',
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['v_proj'],
        layers_to_transform=[0, 1, 2, 3, 4],
        # layers_to_transform=[0, 1, 2, 3, 4, 5, 6, 7],
        bias='none',
    )
LConfigs.append(LConfig7)

# the edge v server and v group for bert
# the edge v server and v group for tiny llama
LConfig8 = LoraConfig(
    # lora config for bert  
        # r=8,
        # lora_alpha=32,
        # target_modules=["value"],
        # layers_to_transform=[0, 1, 2, 3, 8, 9, 10, 11],
        # lora_dropout=0.05,
        # bias="none",
        # task_type="SEQ_CLS"
    # lora config for tiny llama
        task_type='CAUSAL_LM',
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['v_proj'],
        layers_to_transform=[17, 18, 19, 20, 21],
        # layers_to_transform=[24, 25, 26, 27, 28, 29, 30, 31],
        bias='none',
    )
LConfigs.append(LConfig8)


LConfig9 = LoraConfig(
    # lora config for bert  
        # r=8,
        # lora_alpha=32,
        # target_modules=["value"],
        # layers_to_transform=[0, 1, 2, 3],
        # lora_dropout=0.05,
        # bias="none",
        # task_type="SEQ_CLS"
    # lora config for tiny llama
        task_type='CAUSAL_LM',
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        # target_modules=['v_proj'],
        target_modules=['o_proj'],
        layers_to_transform=[0, 1, 2, 3, 4, 17, 18, 19, 20, 21],
        # layers_to_transform=[0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31],
        bias='none',
    )
LConfigs.append(LConfig9)

LConfig10 = LoraConfig(
    # lora config for bert  
        # r=8,
        # lora_alpha=32,
        # target_modules=["value"],
        # layers_to_transform=[4, 5, 6, 7],
        # lora_dropout=0.05,
        # bias="none",
        # task_type="SEQ_CLS"
    # lora config for tiny llama
        task_type='CAUSAL_LM',
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        # target_modules=['v_proj'],
        target_modules=['o_proj'],
        layers_to_transform=[0, 1, 2, 3, 4],
        # layers_to_transform=[0, 1, 2, 3, 4, 5, 6, 7],
        bias='none',
    )
LConfigs.append(LConfig10)

LConfig11 = LoraConfig(
    # lora config for bert 
        # r=8,
        # lora_alpha=32,
        # target_modules=["value"],
        # layers_to_transform=[8, 9, 10, 11],
        # lora_dropout=0.05,
        # bias="none",
        # task_type="SEQ_CLS"
    # lora config for tiny llama
        task_type='CAUSAL_LM',
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        # target_modules=['v_proj'],
        target_modules=['o_proj'],
        layers_to_transform=[17, 18, 19, 20, 21],
        # layers_to_transform=[24, 25, 26, 27, 28, 29, 30, 31],
        bias='none',
    )
LConfigs.append(LConfig11)

# the edge o server and o group for tiny llama
LConfig12 = LoraConfig(
    # lora config for bert 
        # r=8,
        # lora_alpha=32,
        # target_modules=["value"],
        # layers_to_transform=[8, 9, 10, 11],
        # lora_dropout=0.05,
        # bias="none",
        # task_type="SEQ_CLS"
    # lora config for tiny llama
        task_type='CAUSAL_LM',
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['o_proj'],
        # layers_to_transform=[0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20, 21],
        layers_to_transform=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        bias='none',
    )
LConfigs.append(LConfig12)

LConfig13 = LoraConfig(
    # lora config for bert  
        # r=8,
        # lora_alpha=32,
        # target_modules=["value"],
        # layers_to_transform=[0, 1, 2, 3],
        # lora_dropout=0.05,
        # bias="none",
        # task_type="SEQ_CLS"
    # lora config for tiny llama
        task_type='CAUSAL_LM',
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['o_proj'],
        # layers_to_transform=[0, 1, 2, 3, 4, 5, 6],
        layers_to_transform=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        bias='none',
    )
LConfigs.append(LConfig13)

LConfig14 = LoraConfig(
    # lora config for bert  
        # r=8,
        # lora_alpha=32,
        # target_modules=["value"],
        # layers_to_transform=[4, 5, 6, 7],
        # lora_dropout=0.05,
        # bias="none",
        # task_type="SEQ_CLS"
    # lora config for tiny llama
        task_type='CAUSAL_LM',
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['o_proj'],
        # layers_to_transform=[7, 8, 9, 10, 11, 12, 13, 14],
        layers_to_transform=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        bias='none',
    )
LConfigs.append(LConfig14)

LConfig15 = LoraConfig(
    # lora config for bert 
        # r=8,
        # lora_alpha=32,
        # target_modules=["value"],
        # layers_to_transform=[8, 9, 10, 11],
        # lora_dropout=0.05,
        # bias="none",
        # task_type="SEQ_CLS"
    # lora config for tiny llama
        task_type='CAUSAL_LM',
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['o_proj'],
        # layers_to_transform=[15, 16, 17, 18, 19, 20, 21],
        layers_to_transform=[22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        bias='none',
    )
LConfigs.append(LConfig15)

# ===== Define and parse arguments =====
@dataclass
class Arguments:
    # LLM arguments
    base_model_path: Optional[str] = field(default="/root/nfs/LLMDatasetsAndModels/models/TinyLlama-1.1B-intermediate-step-1431k-3T", metadata={"help": "the path of base model"})
    dataset_path: Optional[str] = field(default="/root/nfs/LLMDatasetsAndModels/datasets/medalpaca/medical_meadow_medical_flashcards", metadata={"help": "the path of dataset"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    k_group_clients: Optional[List[int]] = field(default_factory=lambda:[0,1,2], metadata={"help": "The client group used to train the key matrix. The first one is the key edge server"})
    q_group_clients: Optional[List[int]] = field(default_factory=lambda:[3,4,5], metadata={"help": "The client group used to train the query matrix. The first one is the query edge server"})
    v_group_clients: Optional[List[int]] = field(default_factory=lambda:[6,7,8], metadata={"help": "The client group used to train the value matrix. The first one is the value edge server"})
    o_group_clients: Optional[List[int]] = field(default_factory=lambda:[9,10,11], metadata={"help": "The client group used to train the output matrix. The first one is the output edge server"})

    # FL arguments
    num_rounds: Optional[int] = field(default=20, metadata={"help": "the number of global aggregation rounds"})
    num_clients: Optional[int] = field(default=12, metadata={"help": "the number of clients"})
    edge_aggregate_round: Optional[int] = field(default=1, metadata={"help": "the number of edge aggregation rounds"})
    split_strategy: Optional[str] = field(default="assign", metadata={"help": "the split strategy"})
    data_sample: Optional[int] = field(default=33955, metadata={"help": "the data sample number of total client"})
    data_distribution: Optional[List[int]] = field(default_factory=lambda:[3770,3770,3770,3770,3770,3770,1885,1885,1885,1885,1885,1885], metadata={"help": "The client group used to train the query matrix. The first one is the query edge server"})
    seed: Optional[int] = field(default=2024, metadata={"help": "the seed to use"})

# ===== get lora config and some other arguments =====
def get_config(*args):
    if args:
        return LConfigs[args[0]]
    else:
        parser = HfArgumentParser((Arguments,))
        training_args, = parser.parse_args_into_dataclasses()
        return training_args
    

# ===== Define the training arguments =====
def get_training_args(training_configs):
    model_name = os.path.basename(training_configs.base_model_path)
    if model_name in ["bert-base-cased"]:
        training_args = TrainingArguments(
            output_dir=training_configs.output_dir,
            num_train_epochs=2,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=training_args.output_dir,
            logging_steps=150,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            max_grad_norm=1.0
        )
    elif model_name in ["Llama-2-7b-hf", "TinyLlama-1.1B-intermediate-step-1431k-3T"]:
        training_args = TrainingArguments(
            output_dir=training_configs.output_dir,
            per_device_train_batch_size=16,
            learning_rate=5e-5,
            logging_steps=10,
            num_train_epochs=2,
            # max_steps=10,
            report_to=None,
            evaluation_strategy="epoch",
            gradient_checkpointing=True,
            lr_scheduler_type="constant",
            
 
        )
    return training_args

def get_trainer(training_configs,local_model,training_args,train_dataset,eval_dataset):
    model_name = os.path.basename(training_configs.base_model_path)
    if model_name in ["bert-base-cased"]:
        trainer = Trainer(
                model=local_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
    elif model_name in ["Llama-2-7b-hf", "TinyLlama-1.1B-intermediate-step-1431k-3T"]:
        # ===== Define the tokenizer =====
        tokenizer = AutoTokenizer.from_pretrained(training_configs.base_model_path, padding_side="right")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token   # following vicuna

        # ===== Define the formatting function (cater to TRL SFTTrainer)=====
        formatting_prompts_func, response_template = get_formatting_prompts_func("alpaca", tokenizer.eos_token)
        response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
        data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
        trainer = SFTTrainer(
            model=local_model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=512,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
        )
    return trainer

def save_config(training_args):
    now_time = (datetime.now()).strftime("%Y%m%d%H%M%S")
    dataset_name_split = os.path.basename(training_args.dataset_path)
    model_name_split = os.path.basename(training_args.base_model_path)
    output_dir = f"{training_args.output_dir}/{dataset_name_split}_{model_name_split}_{training_args.num_rounds}_{now_time}"
    while True:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            break
        else:
            now_time = (datetime.now() + timedelta(seconds=1)).strftime("%Y%m%d%H%M%S")
            output_dir = f"{training_args.output_dir}/{dataset_name_split}_{model_name_split}_{training_args.num_rounds}_{now_time}"
    training_args.output_dir = output_dir

    # convert arguments to json and save it
    with open(os.path.join(training_args.output_dir, "args.json"), "w") as f:
        combined_dict = {
            "training_args": asdict(training_args),
        }
        json.dump(combined_dict, f, indent=4)


"""
To support TRL supervised fine-tuning. Right now, we need to manually set the template here.
"""

alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response: {}{}"""

vicuna_template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT: {}{}"""

TEMPLATE_DICT = {
    'alpaca': (alpaca_template, '\n### Response:'),
    'vicuna': (vicuna_template, ' ASSISTANT:'),
}

def get_formatting_prompts_func(template_name, eos_token):
    overall_temp, response_temp = TEMPLATE_DICT[template_name]
    def formatting_prompts_func(example):    
        output_texts = []    
        for i in range(len(example['instruction'])):    
            text = overall_temp.format(example['instruction'][i], example['response'][i], eos_token)    
            output_texts.append(text)    
        return output_texts    
    
    return formatting_prompts_func, response_temp