# -*- coding: utf-8 -*-

import pandas as pd
from trainer import Trainer
import os
import argparse

parser = argparse.ArgumentParser(description='data segmentation strategies')
parser.add_argument('--gnn_type', default = "GIN", choices = ["GIN", "GSAGE"], type=str, help='gnn type')
parser.add_argument('--gnn_type_interaction', default = "GIN", choices = ["GIN", "GSAGE"], type=str, help='gnn type interaction')
parser.add_argument('--split_type', default = "discovery", choices = ["discovery", "blind_scaffold", "blind_cell"], type=str, help='split type')
parser.add_argument('--num_folds', default = 10, choices = [10, 20, 5], type=int, help='number of folds')
parser.add_argument('--pos_embedd', default = 1, choices=[0, 1], type=int, help='use position embedding or not')
parser.add_argument('--checkpoint_path', default = '', type=str, help='checkpoint path')
parser.add_argument('--drop_out', default = 0.5, type=float, help='drop out')
parser.add_argument('--gnn_drop_out', default = 0.1, type=float, help='gnn drop out')
parser.add_argument('--gnn_edge_drop_out', default = 0.0, type=float, help='gnn edge drop out')
parser.add_argument('--hidden_dim_2', default = 256, type=int, help='hidden dim 2')
parser.add_argument('--omics', default = "expr + mutation + methylation", type=str, choices=["expr + mutation + methylation", 'expr + mutation', 'expr + methylation', 'mutation + methylation'], help='omics')
args = parser.parse_args()


# optimize params
batch_size = 64
lr =  5e-5
l2_norm = 1e-6
epoch = 100


config_data = {"wash_smiles":True, "brics":False,
        "break_rings":False}


for i in range(1,args.num_folds+1):
    print("-"*40)
    print(f"Training for fold {i}")

    config = {
        'split_type':args.split_type,
        'model_type': "regression", # classification, regression
        'screening': 'None', 
        'external_dataset': 'None',
        'num_layer':4,
        'num_layer_2':4,
        'cell_out_dim':256,
        'hidden_dim':256,
        'hidden_dim_2':args.hidden_dim_2,
        'gnn_type':args.gnn_type,
        'gnn_type_interaction':args.gnn_type_interaction,
        'fold':i,
        'num_folds':args.num_folds, 
        'use_encoder':3,
        'use_scaler':False,
        'checkpoint_path': args.checkpoint_path,
        'batch_size':batch_size,
        'num_workers':4,
        'lr':lr,
        # 'decay':1e-5, #0.00001 (default)
        'decay':l2_norm,
        'train_epoch':epoch,
        'early_stop_epoch':10,
        'drop_out':args.drop_out,
        'gnn_drop_out':args.gnn_drop_out,
        "gnn_edge_drop_out":args.gnn_edge_drop_out,
        'warm_start':3,
        'use_scheduler':True,
        "omics": args.omics, #'expr + mutation + methylation',
        "train_val": False,
        'train_all': False
    }
    config.update(config_data)  

    config['result_folder']=f"./split_method/{config['model_type']}/{config['split_type']}" # for saving fold
    
    config['modeldir'] = "results"
 
    if config["model_type"] == 'regression':
        config["original_dataset_path"] = './data/GDSC/data_processed/CDR_n156813.txt'
    else:
        raise ValueError("Invalid model_type. Choose 'regression'.")

    if os.path.exists(os.path.join(config['modeldir'], f"fold{i}", "test_markdowntable.txt")):
        print("continue...")
        continue
        
    print(config)
    net = Trainer(**config)

    if 'checkpoint_path' in config and config['checkpoint_path']!='': 
        net.load_matching_weights(config['checkpoint_path'])
    net.train()
    print(f"End training for fold {i}")
    print("-"*40)
                


