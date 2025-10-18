from utils import Mol_Tokenizer,get_dist_matrix,get_adj_matrix,molgraph_rep
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
import pandas as pd
import tqdm
import os
import collections
from UniMol import UniMolRepr
from functools import partial
from multiprocessing import Process,Queue,cpu_count,Lock
import lmdb
import pickle
import gc
import time

def process_single_smiles(smi_list, tokenizer_path, result_queue):
    """处理单个分子"""
    tokenizer = Mol_Tokenizer(tokenizer_path)  # 初始化一次
    # unimol_encoder
    unimol = UniMolRepr(data_type='molecule',
                        remove_hs=False,
                        model_name='unimolv1',
                        model_size='84m',
                        use_gpu=True)
    batch_results = {}  # 存储当前批次的计算结果
    smi_batch = []
    batch_size = 64  # **每 10000 个 SMILES 批量存入 LMDB**

    for smi in tqdm.tqdm(smi_list, desc='SMILES Processing...'):
        smi_batch.append(smi)

        if len(smi_batch) >= batch_size:
            try:
                # 批量计算
                batch_num_list, batch_edges, batch_cliques = zip(*[tokenizer.tokenize(s) for s in smi_batch])
                unimol_output = unimol.get_repr(smi_batch, return_atomic_reprs=True)

                for index, s in enumerate(smi_batch):
                    result = {
                        "unimol_embeds": unimol_output['atomic_reprs'][index],
                        "unimol_embedds_mask": unimol_output['atomic_mask'][index],
                        'adj_matrix': get_adj_matrix(batch_num_list[index], batch_edges[index]),
                        'dist_matrix': get_dist_matrix(batch_num_list[index], batch_edges[index]),
                        'nums_list': batch_num_list[index],
                        'cliques': batch_cliques[index],
                        'edges': batch_edges[index],
                        'single_dict': molgraph_rep(s, batch_cliques[index])
                    }
                    batch_results[s] = result  # 先将结果存入本地的字典

            except Exception as e:
                print(f"Error processing {smi}: {e}")
                for smi in smi_batch:
                    if smi not in batch_results:
                        batch_results[smi] = {}

            try:
                result_queue.put(batch_results)
                batch_results = {}  # 清空本地缓存
            except Exception as e:
                print(f"[Queue Error] Failed to put batch into queue: {e}")

            smi_batch = []  # 清空batch
        torch.cuda.empty_cache()  # ✅ 释放显存
        gc.collect()

    # 写入剩余数据
    if smi_batch:
        try:
            batch_num_list, batch_edges, batch_cliques = zip(*[tokenizer.tokenize(s) for s in smi_batch])
            unimol_output = unimol.get_repr(smi_batch, return_atomic_reprs=True)

            for index, smi in enumerate(smi_batch):
                result = {
                    "unimol_embeds": unimol_output['atomic_reprs'][index],
                    "unimol_embedds_mask": unimol_output['atomic_mask'][index],
                    'adj_matrix': get_adj_matrix(batch_num_list[index], batch_edges[index]),
                    'dist_matrix': get_dist_matrix(batch_num_list[index], batch_edges[index]),
                    'nums_list': batch_num_list[index],
                    'cliques': batch_cliques[index],
                    'edges': batch_edges[index],
                    'single_dict': molgraph_rep(smi, batch_cliques[index])
                }
                batch_results[smi] = result

        except Exception as e:
            print(f"[Error] Last batch processing failed: {e}")
            for smi in smi_batch:
                if smi not in batch_results:
                    batch_results[smi] = {}

        try:
            result_queue.put(batch_results)
        except Exception as e:
            print(f"[Queue Error] Failed to put final batch: {e}")

    gc.collect()
    time.sleep(1)  # **避免进程竞争**


def save_dict_to_lmdb(result_queue, lmdb_path, lock):
    """将数据存储到LMDB字典中"""
    env = lmdb.open(lmdb_path, map_size=1099511627776, sync=True, metasync=True, writemap=True)  # 创建LMDB环境

    while True:
        batch_result = result_queue.get()
        if batch_result is None:
            break

        with lock:
            with env.begin(write=True) as txn:
                for smi, result in batch_result.items():
                    # print(f"SMILES:{smi},dict:{result}")
                    txn.put(smi.encode('utf-8'), pickle.dumps(result))

    env.sync()  # 确保数据持久化
    env.close()
    print(f"数据存储已完成：{lmdb_path}")

    # 重新打开数据库，检查数据是否已经成功写入
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        with txn.cursor() as cursor:
            for key, value in cursor:
                value = pickle.loads(value)
                print(f"📖 读取 SMILES: {key.decode('utf-8')}, Map Dict Keys: {value.keys()}成功！")

    env.close()


def calc_all_molecule_dict(data_path, lmdb_path, num_workers=4):
    """将所有SMILES并存入LMDB"""
    pretrain_data = pd.read_csv(data_path)
    pretrain_smi = pretrain_data.iloc[:, 0].values.tolist()
    pretrain_smi_length = len(pretrain_smi)

    tokenizer_path = '../Data/vocab_token.json'
    # fintune_tokenizer_path = r'.\multi_datasets\vocab_token.json'
    result_queue = Queue()  # 进程间通信队列
    lock = Lock()  # **保证多进程存储时不冲突**

    processes = []
    smi_list = []
    index = 0
    for i in range(num_workers):
        batch_smi = pretrain_smi_length // num_workers
        smi_list.append(pretrain_smi[index:index + batch_smi])
        index += batch_smi
    if index < pretrain_smi_length:
        smi_list[-1].extend(pretrain_smi[index:])

    # 启动计算进程
    for j in range(num_workers):
        p = Process(target=process_single_smiles, args=(smi_list[j], tokenizer_path, result_queue))
        processes.append(p)
        p.start()

    # 启动LMDB进程
    write_process = Process(target=save_dict_to_lmdb, args=(result_queue, lmdb_path, lock))
    write_process.start()

    for p in processes:
        p.join()

    # all_molecule_dict = {}
    # with tqdm.tqdm(total=pretrain_smi_length,desc='Processing SMILES') as pbar:
    #     for _ in range(pretrain_smi_length):
    #         smi,result = result_queue.get()
    #         all_molecule_dict[smi] =  result
    #         pbar.update(1)

    # for p in processes:
    #     p.join()
    #
    # save_dict_to_lmdb(all_molecule_dict,lmdb_path)
    # **发送结束信号，通知写入进程退出**
    result_queue.put(None)
    write_process.join()


if __name__ == '__main__':
    # data_path = '/HOME/scz0bnb/run/project_2/My_model_V3/pretrain_datasets/pre_pretrain_data_random.csv'
    # lmdb_path = '/HOME/scz0bnb/run/project_2/My_model_V3/pretrain_datasets/all_molecule_dict_unimol.lmdb'
    fintune_path = r'../Data/MTL_data/multitask_datasets.csv'
    fintune_lmdb_path = r'../Data/all_molecule_dict_unimol.lmdb'
    calc_all_molecule_dict(fintune_path, fintune_lmdb_path)