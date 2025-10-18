import warnings
warnings.warn('ignore')
import numpy as np
from rdkit.Chem import MolFromSmiles,MolToSmiles
from features import (atom_features,bond_features,
                        one_of_k_encoding, one_of_k_encoding_unk)

degrees = [0, 1, 2, 3, 4, 5, 6]
class MolGraph(object):
    def __init__(self):
        #键是节点类型（例如 'atom' 或 'bond'），值是一个列表，包含该类型的节点对象。
        self.nodes = {} # dict of lists of nodes, keyed by node type
    #@ps.snoop()
    def new_node(self, ntype, features=None, rdkit_ix=None):
        new_node = Node(ntype, features, rdkit_ix)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node

    def add_subgraph(self, subgraph):
        """将一个子图（subgraph）合并到当前分子图中"""
        old_nodes = self.nodes
        new_nodes = subgraph.nodes
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))# gain node

    def sort_nodes_by_degree(self, ntype):
        """根据节点的度数（邻居数量）对节点进行排序"""
        nodes_by_degree = {i : [] for i in degrees}
        for node in self.nodes[ntype]:
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)

        new_nodes = []
        for degree in degrees:
            cur_nodes = nodes_by_degree[degree]
            self.nodes[(ntype, degree)] = cur_nodes
            new_nodes.extend(cur_nodes)
        self.nodes[ntype] = new_nodes

    def feature_array(self, ntype):
        """获取指定节点类型的特征数组"""
        assert ntype in self.nodes
        return np.array([node.features for node in self.nodes[ntype]])

    def rdkit_ix_array(self):
        """获取原子节点的 RDKit 索引数组"""
        return np.array([node.rdkit_ix for node in self.nodes['atom']])

    def neighbor_list(self, self_ntype, neighbor_ntype):
        """返回一个嵌套列表，表示每个节点的邻居索引列表"""
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n : i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor]
                 for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]

class Node(object):
    #使用 __slots__ 定义节点对象的属性，减少内存占用
    __slots__ = ['ntype', 'features', '_neighbors', 'rdkit_ix']
    def __init__(self,ntype,features,rdkit_ix):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.rdkit_ix = rdkit_ix

    def add_neighbors(self, neighbor_list):
        """为当前节点添加邻居节点"""
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)

    def get_neighbors(self, ntype):
        """获取指定类型的邻居节点"""
        return [n for n in self._neighbors if n.ntype == ntype]

def graph_from_smiles_tuple(smiles_tuple):
    graph_list = [graph_from_smiles(s) for s in smiles_tuple]
    big_graph = MolGraph()
    #将每个子图合并到大图中
    for subgraph in graph_list:
        big_graph.add_subgraph(subgraph)
    # This sorting allows an efficient (but brittle!) indexing later on.
    #按度数对原子节点进行排序
    big_graph.sort_nodes_by_degree('atom')
    return big_graph

def graph_from_smiles(smiles):
    """将一个SMILES字符串转换为分子图"""
    graph = MolGraph()
    # mol = MolFromSmiles(smiles)
    mol = MolFromSmiles(smiles)
    mol = MolFromSmiles(MolToSmiles(mol))
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)
    #创建原子节点
    atoms_by_rd_idx = {}
    for atom in mol.GetAtoms():
        new_atom_node = graph.new_node('atom', features=atom_features(atom), rdkit_ix=atom.GetIdx())
        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node
    #创建键节点并连接原子节点
    for bond in mol.GetBonds():
        atom1_node = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
        atom2_node = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
        new_bond_node = graph.new_node('bond', features=bond_features(bond))
        # #将键节点连接到对应的原子节点，并将原子节点连接到彼此
        new_bond_node.add_neighbors((atom1_node, atom2_node))
        atom1_node.add_neighbors((atom2_node,))

    #创建分子节点并连接所有原子节点，超级节点
    mol_node = graph.new_node('molecule')
    mol_node.add_neighbors(graph.nodes['atom'])
    return graph

def array_rep_from_smiles(smiles):
    """Precompute everything we need from MolGraph so that we can free the memory asap.
    将一个SMILES字符串转换为数组表示"""
    graph = graph_from_smiles(smiles)
    molgraph = MolGraph()
    molgraph.add_subgraph(graph)
    molgraph.sort_nodes_by_degree('atom')
    #提取原子特征、键特征、原子邻居列表和RDKit索引数组
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'), # List of lists.
                'rdkit_ix'      : molgraph.rdkit_ix_array()}  # For plotting only.
    #提取按度数分类的邻居信息，提取出度为1的原子的邻居原子索引以及键索引
    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep

