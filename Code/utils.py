from rdkit import Chem
from util.chemutils import brics_decomp,tree_decomp,get_clique_mol
from mol_graph import *
import networkx as nx
import numpy as np
import pandas as pd
import json

class Mol_Tokenizer():
    def __init__(self,tokens_id_file):
        #词典中包含子结构：索引
        self.vocab = json.load(open(r'{}'.format(tokens_id_file),'r'))
        self.MST_MAX_WEIGHT = 100
        #词汇表大小
        self.get_vocab_size = len(self.vocab.keys())
        #解码器，得到数字：子结构的映射
        self.id_to_token = {value:key for key,value in self.vocab.items()}

    def tokenize(self,smiles):
        ######返回motif_list：每个子结构对应的“motif”标识符列表。edge：子结构之间的连接关系。ids：子结构的原子索引列表。
        mol = Chem.MolFromSmiles(r'{}'.format(smiles))
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))

        #存储原子索引
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx()) # 把原始 index 存到 atomMapNum

        # 分解分子为子结构（cliques）和连接关系（edges）
        ids,edge = brics_decomp(mol)
        if len(edge) <= 1:
            cliques, edges = tree_decomp(mol)
        motif_list = []
        for id_ in ids:
            #子结构对应的SMILES字符串
            _,token_mols = get_clique_mol(mol,id_)
            #词汇表（vocab）中查找token_mols对应的索引
            token_id = self.vocab.get(token_mols)
            if token_id != None:
                motif_list.append(token_id)
            else:
                motif_list.append(self.vocab.get('<unk>'))
        return motif_list,edge,ids

def get_dist_matrix(num_list,edges):
    ######根据给定的节点列表（num_list）和边列表（edges），生成一个距离矩阵,输出子结构之间的最短路径矩阵
    #创建一个无向图对象
    make_graph = nx.Graph()
    #将边列表中的边添加到图中
    make_graph.add_edges_from(edges)
    #初始化距离矩阵，设置对角线元素为0
    dist_matrix = np.zeros((len(num_list),len(num_list)))
    dist_matrix.fill(1e9)
    row, col = np.diag_indices_from(dist_matrix)
    dist_matrix[row,col] = 0
    #获取图中所有节点的索引，并按升序排序
    graph_nodes = sorted(make_graph.nodes.keys())
    #计算图中所有节点对之间的最短路径长度
    #键是起始节点，值是另一个字典，表示该节点到其他所有节点的最短路径长度。
    all_distance = dict(nx.all_pairs_shortest_path_length(make_graph))
    for dist in graph_nodes:
        node_relative_distance = dict(sorted(all_distance[dist].items(),key = lambda x:x[0]))
        #创建临时字典，处理未连接的节点
        temp_node_dist_dict = {i:node_relative_distance.get(i) if \
        node_relative_distance.get(i)!= None else 1e9 for i in graph_nodes} ### 1e9 refers to Chem.GetDistanceMatrix(mol) in rdkit
        temp_node_dist_list = list(temp_node_dist_dict.values())
        #更新距离矩阵
        dist_matrix[dist][graph_nodes] =  temp_node_dist_list
    return dist_matrix.astype(np.float32)

def get_adj_matrix(num_list,edges):
    adj_matrix = np.eye(len(num_list))
    for edge in edges:
        u,v = edge[0],edge[1]
        adj_matrix[u, v] = 1.0
        adj_matrix[v, u] = 1.0
    return adj_matrix


def extract_bondfeatures_of_neighbors_by_degree(array_rep):
    """
    Sums up all bond features that connect to the atoms (sorted by degree)

    Returns:
    ----------

    list with elements of shape: [(num_atoms_degree_0, 6), (num_atoms_degree_1, 6), (num_atoms_degree_2, 6), etc....]

    e.g.:

    >> print [x.shape for x in extract_bondfeatures_of_neighbors_by_degree(array_rep)]

    [(0,), (269, 6), (524, 6), (297, 6), (25, 6), (0,)]

    根据原子的度数聚合键特征，返回三维矩阵，(度数，每个度对应的原子数，每个原子的聚合后邻居键特征)"""
    bond_features_by_atom_by_degree = []
    for degree in degrees:
        bond_features = array_rep['bond_features']
        bond_neighbors_list = array_rep[('bond_neighbors', degree)]
        summed_bond_neighbors = bond_features[bond_neighbors_list].sum(axis=1)
        bond_features_by_atom_by_degree.append(summed_bond_neighbors)
    return bond_features_by_atom_by_degree

def bond_features_by_degree(total_atoms,summed_degrees,degree):
    """返回每个原子的邻居键特征矩阵，已经按照度进行聚合过，维度是（原子数，10）"""
    mat = np.zeros((total_atoms,10),'float32')
    total_num = []
    if degree == 0:
        for i,x in enumerate(summed_degrees[0]):
            mat[i] = x
        return mat
    else:
        for i in range(degree):
            total_num.append(len(summed_degrees[i]))
        total_num = sum(total_num)
        for i,x in enumerate(summed_degrees[degree]):
            mat[total_num + i] = x
        return mat


def molgraph_rep(smi,cliques):
    #返回原子级特征，以及原子与子结构之间的对应关系
    def atom_to_motif_match(atom_order,cliques):
        #####生成原子与子结构（motif）的匹配矩阵，子结构包含该原子则相应位置值为1
        atom_order = atom_order.tolist()
        temp_matrix = np.zeros((len(cliques),len(atom_order)))
        for th,cli in enumerate(cliques):
            for i in cli:
                temp_matrix[th,atom_order.index(i)] = 1
        return temp_matrix

    def get_adj_dist_matrix(mol_graph,smi):
        #####返回调整后的邻接矩阵和距离矩阵
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        num_atoms = mol.GetNumAtoms()
        adjoin_matrix_temp = np.eye(num_atoms)
        adj_matrix = Chem.GetAdjacencyMatrix(mol)
        adj_matrix = (adjoin_matrix_temp + adj_matrix)[:,mol_graph['rdkit_ix']][mol_graph['rdkit_ix']]
        dist_matrix = Chem.GetDistanceMatrix(mol)[:,mol_graph['rdkit_ix']][mol_graph['rdkit_ix']]
        return adj_matrix,dist_matrix

    #初始化特征字典
    single_dict = {'input_atom_features':[],
            'atom_match_matrix':[],
            'sum_atoms':[],
            'adj_matrix':[],
            'dist_matrix':[]
            }
    #将分子图表示为数组
    array_rep = array_rep_from_smiles(smi)
    #返回按度数（degree）聚合的邻居键特征
    summed_degrees = extract_bondfeatures_of_neighbors_by_degree(array_rep)
    atom_features = array_rep['atom_features']
    #提取邻居特征
    all_bond_features = []
    for degree in degrees:
        #返回度为1的原子对应的邻居原子索引
        atom_neighbors_list = array_rep[('atom_neighbors', degree)].astype('int32')
        if len(atom_neighbors_list)==0:
            true_summed_degree = np.zeros((atom_features.shape[0], 10),'float32')
        else:
            # atom_neighbor_matching_matrix = connectivity_to_Matrix(array_rep, atom_features.shape[0],degree)
            true_summed_degree = bond_features_by_degree(atom_features.shape[0],summed_degrees,degree)
        # atom_selects = np.matmul(atom_neighbor_matching_matrix,atom_features)
        # merged_atom_bond_features = np.concatenate([atom_features,true_summed_degree],axis=1)
        all_bond_features.append(true_summed_degree) #是三维矩阵，(度数，总原子数，10)
    #计算每个子结构中包含的原子数量
    single_dict['atom_match_matrix'] = atom_to_motif_match(array_rep['rdkit_ix'],cliques)
    single_dict['sum_atoms'] = np.reshape(np.sum(single_dict['atom_match_matrix'],axis=1),(-1,1))
    out_bond_features = 0
    #out_bond_features：(总原子数，10)得到每个原子的聚合后的邻居键特征，是按照度进行排列的
    for arr in all_bond_features:
        out_bond_features = out_bond_features + arr
    #将原子特征和相应的邻居键特征拼合作为输入，按度进行排列的
    single_dict['input_atom_features'] = np.concatenate([atom_features,out_bond_features],axis=1)
    #生成邻接矩阵和距离矩阵
    adj_matrix,dist_matrix = get_adj_dist_matrix(array_rep,smi)
    single_dict['adj_matrix'] = adj_matrix
    single_dict['dist_matrix'] = dist_matrix
    single_dict = {key:np.array(value,dtype='float32') for key,value in single_dict.items()}
    #返回包含分子图表示的字典
    return single_dict

if __name__ == '__main__':
    smi = 'C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1'
    mol_token = Mol_Tokenizer(r'pretrain_datasets\vocab_token.json')
    token_ids,edges,ids = mol_token.tokenize(smi)
    single_dict = molgraph_rep(smi,ids)
    print(single_dict['input_atom_features'].shape)


