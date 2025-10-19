from argparse import ArgumentParser
from MFPK_model import *
from datasets import GraphBertDataset,collate_fn,move_to_device
from utils import Mol_Tokenizer
from util import *
from save_data_unimol import calc_all_molecule_dict
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

#wandb.login(key="14544ccbbc4008dbcae3715dd8611f3856ff23ca")

def load_pretrain_weights(model,pretrained_weights_path):
    pretrained_dict = torch.load(pretrained_weights_path)
    #过滤掉mlp层的参数
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

def train(model, train_loader, optimizer, criterion, epoch, args):
    model.train()

    header = f'Train Epoch:{epoch}'
    print_feq = 10

    tqdm_data_loader = tqdm.tqdm(train_loader, desc=header, miniters=print_feq)
    metrics = Metrics(args)
    scaler = torch.amp.GradScaler('cuda')
    total_loss = 0

    task_losses_num = torch.zeros(args.task_num, device=args.device)  # 累计每个任务的损失

    if hasattr(args, 'weights') and args.weights is not None:
        task_weights = torch.tensor(args.weights, device=args.device)
    else:
        task_weights = torch.ones(args.task_num, device=args.device)  # 初始化权重

    #保留当前的权重
    old_task_weights = task_weights.clone()

    task_valid_counts = torch.zeros(args.task_num, device=args.device)

    # 只对第一个 batch 使用 Profiler
    for batch_idx, batch in enumerate(tqdm_data_loader):
        start_time = time.time()

        # 正常训练
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

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            _,prediction, constrastive_loss = model(
                unimol_embeds, unimol_embeds_mask, atom_feats, atom_adj_matrix,
                atom_dist_matrix, atom_match_matrix, sum_atoms, motif_num_list,
                motif_adj_matrix, motif_dist_matrix, args
            )
            target = label.clone()
            mask = ~torch.isnan(target).bool()
            task_valid_counts += mask.sum(dim=0)
            target = torch.where(mask, target, torch.tensor(0.0, device=args.device))

            loss_per_task = criterion(prediction, target) * mask
            task_losses_num += loss_per_task.sum(dim=0)  # 累计每个任务的损失
            weighted_loss = loss_per_task * task_weights
            loss = weighted_loss.sum() / mask.sum() + args.weight_ratio * constrastive_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        metrics.update(prediction, label)
        total_loss += loss.item()

        end_time = time.time()
        print(f"One batch needs:{end_time - start_time}")

        with torch.no_grad():  # 确保不保存梯度
            del atom_feats, atom_adj_matrix, atom_dist_matrix, atom_match_matrix, sum_atoms
            del motif_num_list, motif_adj_matrix, motif_dist_matrix, unimol_embeds, unimol_embeds_mask
            del label, target, mask, prediction, loss, constrastive_loss

    ###动态调整权重
    task_losses_avg = task_losses_num / (task_valid_counts + 1e-8)  # 一个epoch每个任务的平均损失
    task_weights = task_losses_avg / (task_losses_avg.mean() + 1e-8)  # 归一化权重
    args.weights = task_weights.tolist()

    fold_error = metrics.fold_error()
    gmfe = metrics.calculate_gmfe()
    afe = metrics.calculate_afe()
    mfe = metrics.median_fold_error()
    bias = metrics.calculate_bias()
    rmse_r2 = metrics.calculate_rmse_r2()
    pearson_r = metrics.calculate_pearson_r()
    total_avg_loss = total_loss / len(train_loader)
    # print('epoch {:d}/{:d}, training {} '.format(
    # epoch, args.epoch, gmfe, fold_error))

    return total_avg_loss, fold_error, gmfe, afe, mfe, bias, rmse_r2, pearson_r, old_task_weights

def valid(model, test_loader, criterion, epoch, args, pre_weights):
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
                pre_weights = pre_weights.to(args.device).unsqueeze(0)
                weighted_loss = loss * pre_weights
                loss = weighted_loss.sum() / mask.sum() + args.weight_ratio * constrastive_loss

            valid_loss += loss.item()
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
    valid_avg_loss = valid_loss / len(test_loader)
    # print('epoch {:d}/{:d}, validation {} '.format( epoch, args.epoch, gmfe))

    return valid_avg_loss, fold_error, gmfe, afe, mfe, bias, rmse_r2, pearson_r


def calculate_metrics(args,fold_error, gmfe, afe, mfe, bias, rmse_r2,pearson_r):
    metrics = [fold_error,gmfe, afe, mfe, bias, rmse_r2, pearson_r]
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
    data_path = '../Data/MTL_data/multitask_datasets.csv'
    total_data = pd.read_csv(data_path)
    shuffled_indices = np.random.permutation(len(total_data))
    total_datasets = total_data.iloc[shuffled_indices].reset_index(drop=True)
    train_size = int(len(total_datasets) * 0.9)
    train_data = total_datasets.iloc[:train_size,:]
    test_data = total_datasets.iloc[train_size:,:]

    #calculate feats
    calc_all_molecule_dict(data_path,args.lmdb_path)

    #提取数据
    # train_data.to_csv('/home/jjfang/multi_datasets_V3/train_multi_datasets.csv',index=False)
    # test_data.to_csv('/home/jjfang/multi_datasets_V3/test_multi_datasets.csv',index=False)

    fold_datasets = []
    kfold = KFold(n_splits=5,shuffle=True,random_state=seed)
    for train_index,valid_index in kfold.split(train_data):
        train_fold_data = train_data.iloc[train_index]
        valid_fold_data = train_data.iloc[valid_index]

        train_fold_datasets = GraphBertDataset(train_fold_data,tokenizer,args.lmdb_path,args)
        valid_fold_datasets = GraphBertDataset(valid_fold_data,tokenizer,args.lmdb_path,args)
        train_fold_loader = DataLoader(train_fold_datasets,batch_size=args.batch_size,shuffle=True,pin_memory=True,collate_fn=collate_fn)
        valid_fold_loader = DataLoader(valid_fold_datasets,batch_size=args.batch_size,shuffle=True,pin_memory=True,collate_fn=collate_fn)
        fold_datasets.append((train_fold_loader,valid_fold_loader))

    test_datasets = GraphBertDataset(test_data,tokenizer,args.lmdb_path,args)
    test_loader = DataLoader(test_datasets,batch_size=args.batch_size,shuffle=False,pin_memory=True,collate_fn=collate_fn)

    #model
    model = MFPK(config).to(args.device)
    # 冻结unimol参数
    # for param in model.unimol.model.parameters():
    #     param.requires_grad = False
    # 计算模型总参数
    print('#parameters:', sum(p.numel() for p in model.parameters()))

    #导入预训练权重
    if args.model_path:
        load_pretrain_weights(model,args.model_path)

    arg_opt = config['optimizer']
    optimizer = optim.AdamW(model.parameters(),lr=arg_opt['lr'],weight_decay=arg_opt['weight_decay'])
    loss = nn.MSELoss(reduction='none')
    scheduler = CosineAnnealingLR(optimizer,T_max=config['schedular']['epochs'],eta_min=config['schedular']['min_lr'])

    os.makedirs(f"Result",exist_ok=True)
    for index,(train_loader,valid_loader) in enumerate(fold_datasets):

        # 初始化
        wandb.init(project="MMPK_V3_{index}", mode='offline')

        best_valid = 10000.
        best_epoch = 0
        patience = 0
        stop_time = 20

        train_outcome = []
        valid_outcome = []
        for epoch in range(args.epoch):
            print('Epoch', epoch)
            total_loss,fold_error,gmfe,afe,mfe,bias,rmse_r2,train_pearson,weights = train(model,train_loader,optimizer,loss,epoch,args)
            train_res = calculate_metrics(args, fold_error, gmfe, afe, mfe, bias, rmse_r2,train_pearson)
            train_outcome.append(train_res)

            valid_loss, fold_error, gmfe, afe, mfe, bias, rmse_r2,valid_pearson = valid(model, valid_loader,loss, epoch, args, weights)
            valid_res = calculate_metrics(args, fold_error, gmfe, afe, mfe, bias, rmse_r2,valid_pearson)
            valid_outcome.append(valid_res)
            wandb.log({"Epoch": epoch, "Train_Loss": total_loss, "Test_Loss": valid_loss})

            scheduler.step()

            # 记录当前的学习率
            current_lr = scheduler.get_last_lr()[0]
            wandb.log({'Lr':current_lr,'Epoch':epoch})

            # early stopping
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_epoch = epoch
                patience = 0  # 重置 patience
                best_weights = weights
                torch.save(model.state_dict(), f"Model/Finetune/fintune_{index}_fold_model.pt")
            else:
                patience += 1
            if patience > stop_time:
                break

        model.load_state_dict(torch.load(f"Model/Finetune/fintune_{index}_fold_model.pt"))
        _, fold_error, gmfe, afe, mfe, bias, rmse_r2,test_pearson = valid(model, test_loader, loss, best_epoch, args, best_weights)
        test_res = calculate_metrics(args, fold_error, gmfe, afe, mfe, bias, rmse_r2, test_pearson)

        train_outcome[best_epoch].to_csv(f"Result/fintune_{index}_train.csv",index=False)
        valid_outcome[best_epoch].to_csv(f"Result/fintune_{index}_valid.csv",index=False)
        test_res.to_csv(f"Result/fintune_{index}_test.csv",index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed',default=42,type=int)
    parser.add_argument('--epoch',default=200,type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--min_lr',default=3e-6, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--model_path',default='../Model/Pretrain/supervised_pretrain_model.pt',type=str)
    parser.add_argument('--lmdb_path',default='../Data/all_molecule_dict_unimol.lmdb',
                        type=str)
    parser.add_argument('--weight_ratio',default=0.1,type=float)
    parser.add_argument('--task_num',default=17,type=int)
    parser.add_argument('--task_names', default=[ 'human_VDss_L_kg', 'human_CL_mL_min_kg', 'human_fup','human_mrt', 'human_thalf',
                          'dog_VDss_L_kg', 'dog_CL_mL_min_kg', 'dog_fup','monkey_VDss_L_kg', 'monkey_CL_mL_min_kg', 'monkey_fup',
                          'rat_VDss_L_kg','rat_CL_mL_min_kg', 'rat_fup','mouse_VDss_L_kg','mouse_CL_mL_min_kg', 'mouse_fup'])
    parser.add_argument('--metrics',
                        default=['Two fold error', 'Three fold error', 'Five fold error', 'GMFE', 'AFE', 'MFE', 'Bias', 'RMSE',
                                 'R2', 'Pearson_r'])
    args = parser.parse_args()

    #param settings
    tokenizer = Mol_Tokenizer('../Data/vocab_token.json')
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

    main(args,pretrain_config,tokenizer)

