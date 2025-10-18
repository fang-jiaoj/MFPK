#from utils import Mol_Tokenizer,get_dist_matrix,get_adj_matrix,molgraph_rep
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
import pandas as pd
import tqdm
import os
import collections
from functools import partial
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import lmdb
import pickle
import gc
import time

class GraphBertDataset(Dataset):
    """自定义的Dataset对象"""
    def __init__(self,dataset,tokenizer,lmdb_path,args):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.pad_value = tokenizer.vocab['<pad>']
        #包含所有分子对应的特征，无论训练集还是测试集
        self.lmdb_path = lmdb_path
        self.args = args
        #self.env = None  # 延迟初始化
        self.env = lmdb.open(self.lmdb_path,readonly=True,lock=False,max_readers=1024) #打开LMDB环境

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        item = self.dataset.iloc[idx] #series
        smi,label = item[0],item[1:]
        #smi = item[0]
        return self.numerical_seq(smi,label)

    # def worker_init_fn(self, worker_id):
    #     """Worker初始化函数"""
    #     if torch.cuda.is_available():
    #         self.args.device = torch.device("cuda:0")
    #     else:
    #         raise RuntimeError("CUDA not available in worker")

    def numerical_seq(self,smi,label):
        """从LMDB中加载数据,不能一个一个放在GPU上会导致极慢"""
        with self.env.begin() as txn:
            key = smi.encode('utf-8')
            value = txn.get(key)
            if value is None:
                raise  KeyError(f"SMILES {smi} not found in LMDB")
            map_dict = pickle.loads(value)

        # # 直接在 GPU 上创建张量
        # device = getattr(self.args, 'device', torch.device('cuda:0'))

        #提取motif级特征
        nums_list = torch.tensor([self.tokenizer.vocab['<global>']] + map_dict['nums_list'],dtype=torch.long) #数值化的motif列表
        adjoin_matrix = torch.ones((len(nums_list),len(nums_list)),dtype=torch.float32)
        adjoin_matrix[1:,1:] = torch.tensor(map_dict['adj_matrix'])
        adjoin_matrix = (1 - adjoin_matrix) * -1e9

        dist_matrix = torch.ones((len(nums_list),len(nums_list)),dtype=torch.float32)
        dist_matrix[1:,1:] = torch.tensor(map_dict['dist_matrix'])

        #额外特征，二维列表
        cliques = map_dict['cliques']

        #提取原子级特征
        single_dict_atom = map_dict['single_dict']
        atom_features = torch.tensor(single_dict_atom['input_atom_features'],dtype=torch.float32)
        dist_matrix_atom = torch.tensor(single_dict_atom['dist_matrix'],dtype=torch.float32)
        adjoin_matrix_atom = torch.tensor(single_dict_atom['adj_matrix'],dtype=torch.float32)
        adjoin_matrix_atom = (1 - adjoin_matrix_atom) * (-1e9)
        atom_match_matrix = torch.tensor(single_dict_atom['atom_match_matrix'],dtype=torch.float32)
        sum_atoms = torch.tensor(single_dict_atom['sum_atoms'],dtype=torch.float32)

        #提取unimol级别特征
        unimol_embeds = torch.tensor(map_dict["unimol_embeds"],dtype=torch.float32)
        unimol_embeds_mask = torch.tensor(map_dict["unimol_embedds_mask"],dtype=torch.float32)

        #掩码motif token
        # masked_motif_indicies = np.random.permutation(len(nums_list)-1)[:max(int(len(nums_list)*0.15),1)] + 1
        # y = np.array(nums_list).astype("int64")
        # weight = np.zeros(len(nums_list))
        # for i in masked_motif_indicies:
        #     rand = np.random.rand()
        #     weight[i] = 1
        #     if rand < 0.8:
        #         nums_list[i] = self.tokenizer.vocab['<mask>']
        #     elif rand < 0.9:
        #         nums_list[i] = int(np.random.rand() * 11890 + 1)
        # weight = weight.astype("float32")
        #print("X:",len(x))

        return {
            'SMILES': smi,
            'cliques':cliques,
            'x': nums_list,
            'adjoin_matrix': adjoin_matrix,
            'dist_matrix': dist_matrix,
            'atom_features': atom_features,
            'adjoin_matrix_atom': adjoin_matrix_atom,
            'dist_matrix_atom': dist_matrix_atom,
            'atom_match_matrix': atom_match_matrix,
            'sum_atoms': sum_atoms,
            'unimol_embeds': unimol_embeds,
            'unimol_embeds_mask': unimol_embeds_mask,
            'label': torch.tensor(label, dtype=torch.float32)
        }


def collate_fn(batch):
    """将单个样本处理成batch数据"""
    def pad_batch_data(batch_data,pad_value):
        max_row = max(item.size(0) for item in batch_data)
        max_col = max(item.size(1) for item in batch_data)

        device = batch_data[0].device
        pad_matrix = torch.full((len(batch_data),max_row,max_col),pad_value,device=device)
        for index,item in enumerate(batch_data):
            pad_matrix[index,:item.size(0),:item.size(1)] = item

        return pad_matrix

    #得到键为特征，值是一个batch的特征值
    batch = {k: [item[k] for item in batch] for k in batch[0].keys()}
    #torch.nn.utils.rnn.pad_sequence将一个batch序列填充到相同的长度,以该batch中最长的长度为准
    #(batch_size,最长的子结构数),pad_sequence()函数要求输入的张量是二维张量
    batch['x'] = pad_sequence(batch['x'],batch_first=True,padding_value=0)
    batch['adjoin_matrix'] = pad_batch_data(batch['adjoin_matrix'],0)
    batch['dist_matrix'] =pad_batch_data(batch['dist_matrix'],-1e9)
    batch['atom_features'] = pad_batch_data(batch['atom_features'],0)
    batch['adjoin_matrix_atom'] = pad_batch_data(batch['adjoin_matrix_atom'],0)
    batch['dist_matrix_atom'] = pad_batch_data(batch['dist_matrix_atom'],-1e9)
    batch['atom_match_matrix'] = pad_batch_data(batch['atom_match_matrix'],0).to(torch.float32)
    batch['sum_atoms'] = pad_sequence(batch['sum_atoms'],batch_first=True,padding_value=1).to(torch.float32)
    batch['unimol_embeds'] = pad_batch_data(batch['unimol_embeds'],0).to(torch.float32)
    batch['unimol_embeds_mask'] = pad_sequence(batch['unimol_embeds_mask'],padding_value=1)

    #batch['label']填充，则无法计算mask
    return batch

def move_to_device(batch,device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}

def generate_scaffold(smiles):
    """获取分子的骨架"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return scaffold

def scaffold_split(data,smiles_column='SMILES',ratio=0.1,seed=42):
    """将数据集按照ratio进行骨架划分"""
    scaffold_to_indicies = defaultdict(list)

    #{scaffold:[index]}
    for index,smi in enumerate(data[smiles_column]):
        scaffold = generate_scaffold(smi)
        if scaffold is not None:
            scaffold_to_indicies[scaffold].append(index)

    #sorted by size (larger scaffold groups first) for balanced split
    sorted_scaffolds = sorted(scaffold_to_indicies.items(), key=lambda x: len(x[1]),reverse=True)

    #为了使得标签分配均匀
    train_indices,test_indices = [],[]
    total_sample = len(data)
    test_cutoff = int(total_sample * ratio)

    for scaffold,indices in sorted_scaffolds:
        if len(indices) + len(test_indices) <= test_cutoff:
            test_indices.extend(indices)
        else:
            train_indices.extend(indices)

    train_datasets = data.iloc[train_indices].reset_index(drop=True)
    test_datasets = data.iloc[test_indices].reset_index(drop=True)
    return train_datasets,test_datasets

















