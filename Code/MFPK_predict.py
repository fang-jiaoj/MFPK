from argparse import ArgumentParser
from MFPK_model import *
from datasets import GraphBertDataset,collate_fn,move_to_device
from utils import Mol_Tokenizer
from util import *
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
import torch.optim as optim
from calc_metrics import Metrics
from collections import defaultdict
from sklearn.model_selection import KFold
from itertools import product
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import random
import tqdm
import wandb
import os
import gc
import time

def load_pretrain_weights(model,pretrained_weights_path):
    pretrained_dict = torch.load(pretrained_weights_path)
    #全参微调
    pretrained_dict = {k: v for k,v in pretrained_dict.items()}

    model_dict = model.state_dict()
    filtered_dict = {}

    for k,v in pretrained_dict.items():
        if k in model_dict and model_dict[k].size() == v.size():
            filtered_dict[k] = v
        else:
            model_shape = model_dict[k].shape if k in model_dict else "Not found"
            print( f"Skipped loading parameter: {k} | checkpoint shape: {v.shape}, model shape: {model_shape}")

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict,strict=False)

def valid(model, test_loader, criterion, args):
    model.eval()
    valid_metrics = Metrics(args)

    valid_loss = 0
    for index, batch in enumerate(test_loader):
        with torch.no_grad():
            batch = move_to_device(batch,args.device)

            smi = batch['SMILES']
            atom_feats = batch['atom_features']
            atom_adj_matrix = batch['adjoin_matrix_atom']
            atom_dist_matrix = batch['dist_matrix_atom']
            atom_match_matrix = batch['atom_match_matrix']
            sum_atoms = batch['sum_atoms']

            motif_num_list = batch['x']
            motif_adj_matrix = batch['adjoin_matrix']
            motif_dist_matrix = batch['dist_matrix']

            unimol_embeds = batch['unimol_embeds']
            unimol_embeds_mask = batch['unimol_embeds_mask']
            label = torch.stack(batch['label'], dim=0).to(args.device)

            with torch.amp.autocast('cuda'):
                _,prediction, constrastive_loss = model(
                    unimol_embeds, unimol_embeds_mask, atom_feats, atom_adj_matrix,
                    atom_dist_matrix, atom_match_matrix, sum_atoms, motif_num_list,
                    motif_adj_matrix, motif_dist_matrix, args
                )
                target = label.clone()
                mask = ~torch.isnan(target).bool()
                target = torch.where(mask, target, torch.tensor(0.0, device=args.device))
                loss = criterion(prediction, target) * mask
                # pre_weights = pre_weights.to(args.device).unsqueeze(0)
                # weighted_loss = loss * pre_weights
                # loss = weighted_loss.sum() / mask.sum() + args.weight_ratio * constrastive_loss

            # valid_loss += loss.item()
            valid_metrics.update(prediction, label)
            # target = torch.Tensor([[0 if np.isnan(x) else x for x in value] for value in target]).to(args.device)
            # target = target.float().to(args.device)

    fold_error = valid_metrics.fold_error()
    gmfe = valid_metrics.calculate_gmfe()
    afe = valid_metrics.calculate_afe()
    mfe = valid_metrics.median_fold_error()
    bias = valid_metrics.calculate_bias()
    rmse_r2 = valid_metrics.calculate_rmse_r2()
    pearson_r = valid_metrics.calculate_pearson_r()

    true_label,predict_label = valid_metrics.multi_datasets()
    #valid_avg_loss = valid_loss / len(test_loader)
    # print('epoch {:d}/{:d}, validation {} '.format( epoch, args.epoch, gmfe))

    return true_label,predict_label, fold_error, gmfe, afe, mfe, bias, rmse_r2, pearson_r


def calculate_metrics(args,fold_error, gmfe, afe, mfe, bias, rmse_r2, pearson_r):
    metrics = [fold_error,gmfe,afe, mfe,bias,rmse_r2,pearson_r]
    merged = defaultdict(list)
    for item in metrics:
        for key,value in item.items():
            if isinstance(value,(list,tuple)):
                merged[key].extend(value)

    merged = pd.DataFrame(merged)
    merged.index = args.metrics
    for key, value in merged.items():
          print(f"Key: {key}, Value type: {type(value)}, Value: {value}")

    return merged

def main(args,config,tokenizer):

    #seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    #split datasets
    #total_data = pd.read_csv('/home/jjfang/multi_datasets/multitask_datasets.csv')
    predict_data = pd.read_csv('../Data/MTL_data/test_multi_datasets.csv')

    test_datasets = GraphBertDataset(predict_data,tokenizer,args.lmdb_path,args)
    test_loader = DataLoader(test_datasets,batch_size=args.batch_size,shuffle=False,pin_memory=True,collate_fn=collate_fn)

    fintune_model_path = "../Model/Finetune/"
    loss = nn.MSELoss(reduction='none')

    test_output_dir = os.path.join("Result/", "test_datasets_prediction")
    os.makedirs(test_output_dir, exist_ok=True)
    res_dict = {task: [] for task in args.task_names}

    for fold in range(5):
        model = MFPK(config).to(args.device)
        args.model_path = os.path.join(fintune_model_path, f'fintune_{fold}_fold_model.pt')
        if args.model_path:
            load_pretrain_weights(model,args.model_path)

            true_label,predict_label,fold_error, gmfe, afe, mfe, bias, rmse_r2, pearson_r = valid(model,test_loader,loss,args)

            test_res = calculate_metrics(args,fold_error, gmfe, afe, mfe, bias, rmse_r2, pearson_r) #所有任务的指标
            test_res.to_csv(os.path.join(test_output_dir, f"fintune_{fold}_test.csv"),index=False)

            for index,task_name in enumerate(args.task_names):
                task_pred_label = predict_label[index]

                res_dict[task_name].append(pd.Series(task_pred_label))

    res_df = pd.DataFrame()
    res_smi = predict_data['SMILES']
    for task_name,data in res_dict.items():
        all_preds = pd.concat(data,axis=1)
        all_preds.columns = [f'fold_{i}_pred' for i in range(len(data))]

        mean_pred = all_preds.mean(axis=1)
        task_df = pd.DataFrame({
            task_name: mean_pred
        })
        res_df = pd.concat([res_df,task_df],axis=1)

    res_df.insert(0,'SMILES',res_smi)
    res_path = os.path.join(test_output_dir, f"test_multi_pred_label.csv")
    res_df.to_csv(res_path, index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed',default=42,type=int)
    parser.add_argument('--epoch',default=200,type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--min_lr',default=3e-6, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    #parser.add_argument('--model_path',default='/home/jjfang/pretrain_res/supervised_pretrain_model.pt',type=str)
    parser.add_argument('--lmdb_path', default='/home/jjfang/multi_datasets_V3/trial_1_all_molecule_dict_unimol.lmdb',
                         type=str)
    #parser.add_argument('--lmdb_path',default='/home/jjfang/external_datasets/external_animal_human_molecule_dict_unimol.lmdb',
                        #type=str)
    parser.add_argument('--weight_ratio',default=0.1,type=float)
    parser.add_argument('--task_num',default=17,type=int)
    parser.add_argument('--task_names', default=[ 'human_VDss_L_kg', 'human_CL_mL_min_kg', 'human_fup',
                                                 'human_mrt', 'human_thalf','dog_VDss_L_kg', 'dog_CL_mL_min_kg', 'dog_fup',
                                                  'monkey_VDss_L_kg', 'monkey_CL_mL_min_kg', 'monkey_fup', 'rat_VDss_L_kg',
                                                  'rat_CL_mL_min_kg', 'rat_fup', 'mouse_VDss_L_kg', 'mouse_CL_mL_min_kg',
                                                  'mouse_fup'])
    parser.add_argument('--metrics',
                        default=['Two fold error', 'Three fold error', 'Five fold error', 'GMFE', 'AFE', 'MFE', 'Bias', 'RMSE',
                                 'R2','Pearson_r'])
    args = parser.parse_args()

    #param settings
    tokenizer = Mol_Tokenizer('/home/jjfang/pretrain_datasets/vocab_token.json')
    pretrain_config = {'median': {'name': 'Median', 'num_layers': 6, 'num_heads': 8, 'd_model': 512,'dff':512},
                       'input_vocab_size':tokenizer.get_vocab_size,'dropout':0.2,'embed_dim':256,'temp':0.2,
                       'task_num':args.task_num,
                       'schedular': {'sched': 'cosine', 'lr': args.lr, 'epochs': args.epoch, 'min_lr': args.min_lr,
                                     'decay_rate': 1, 'warmup_lr': 0.5e-5, 'warmup_epochs': 1, 'cooldown_epochs': 0},
                       'optimizer': {'opt': 'adamW', 'lr': args.lr, 'weight_decay': 1e-5},
                       }

    if torch.cuda.is_available():
        print('cuda:0')
        args.device = torch.device('cuda:0')
    else:
        print('cpu')
        args.device = torch.device('cpu')
    #mp.set_start_method('spawn')  # 关键设置
    main(args,pretrain_config,tokenizer)

