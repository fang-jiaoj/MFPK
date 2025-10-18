from torch.nn.functional import dropout
#from UniMol import UniMolRepr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import lmdb


class AttrDict(dict):
    """AttrDict 的目的是提供一个既可以像普通字典那样使用，也可以像对象属性那样访问的字典，如：config.learning_rate"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def rescale_distance_matrix(w):
    """将距离矩阵 w 的值映射到一个接近于 0 到 1 的区间内"""
    constant_value = torch.tensor(1.0, dtype=torch.float32)  # 默认设置为 1
    numerator = constant_value + torch.exp(constant_value)
    denominator = constant_value + torch.exp(constant_value - w)
    return numerator / denominator

def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

def get_angles(pos, i, d_model):
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / float(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(torch.arange(position).unsqueeze(1),
                            torch.arange(d_model).unsqueeze(0),
                            d_model)
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads.unsqueeze(0)
    return pos_encoding

def scaled_dot_product_attention(q, k, v, mask, adjoin_matrix, dist_matrix):
    """q,k,v,mask,adjoin_mask,dist_matrix都是四维矩阵"""
    # 预计算缩放因子 (避免重复计算)
    dk = torch.tensor(k.size(-1), dtype=torch.float32, device=q.device)
    scale_factor = 1.0 / torch.sqrt(dk)

    if dist_matrix is not None:
        dist_matrix = rescale_distance_matrix(dist_matrix.to(q.device))
        attn_bias = (F.relu(torch.matmul(q,k.transpose(-2,-1))) * dist_matrix) * scale_factor
    else:
        attn_bias = torch.matmul(q,k.transpose(-2,-1)) * scale_factor

    if mask is not None:
        attn_bias = attn_bias + (mask * -1e9)
    if adjoin_matrix is not None:
        attn_bias = attn_bias + adjoin_matrix

    attention_weights = F.softmax(attn_bias, dim=-1)
    output = torch.einsum('bhij,bhjd->bhid', attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model, num_heads):
        super(MultiHeadAttention,self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        self.W_Q = nn.Linear(d_model,d_model)
        self.W_K = nn.Linear(d_model,d_model)
        self.W_V = nn.Linear(d_model,d_model)
        self.fc = nn.Linear(d_model,d_model)

        # 缓存参数 (避免重复初始化)
        self.register_buffer('scale_factor', torch.tensor(1.0 / (d_model  ** 0.5)))

    def forward(self,input_Q,input_K,input_V,mask,adj_matrix=None,dist_matrix=None):
        """input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        mask: [batch_size, 1,1,seq_len]"""

        batch_size = input_Q.size(0)
        ## Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2).contiguous()
        K = self.W_K(input_K).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2).contiguous()
        V = self.W_V(input_V).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2).contiguous()

        ## scaled_attention: [batch_size, n_heads, len_q, d_v], attention_weights: [batch_size, n_heads, len_q, len_k]
        attn_output,attention_weights = scaled_dot_product_attention(Q,K,V,mask,adj_matrix,dist_matrix)
        ## concat_attention: [batch_size, len_q, n_heads * d_v]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc(attn_output)
        return output, attention_weights

def feed_forward_network(d_model, dff):
    return nn.Sequential(
        nn.Linear(d_model, dff),
        nn.GELU(),
        nn.Linear(dff, d_model))

class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,dff,rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(int(d_model/2), num_heads)
        self.mha2 = MultiHeadAttention(int(d_model/2), num_heads)
        self.ffn = feed_forward_network(d_model,dff)
        self.layernorm1 = nn.LayerNorm(d_model,eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(p=rate)
        self.dropout2 = nn.Dropout(p=rate)

    def forward(self,x=None,encoder_padding_mask=None,adj_matrix=None,dist_matrix=None,
                unimol_embeds=None,motif_feats=None,motif_padding_mask=None):
        """# x.shape          : (batch_size, seq_len, dim=d_model)
        # attn_output.shape: (batch_size, seq_len, d_model)
        # out1.shape       : (batch_size, seq_len, d_model)"""

        x1,x2 = torch.split(x,int(x.size(-1) / 2),dim=-1)
        x_l,attention_weights_local = self.mha1(x1,x1,x1,encoder_padding_mask,adj_matrix,dist_matrix=None)
        x_g,attention_weights_gloabl = self.mha2(x2,x2,x2,encoder_padding_mask,adj_matrix= None,dist_matrix=dist_matrix)
        attn_output = torch.cat([x_l,x_g],dim=-1)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(ffn_output + out1)

        return out2,attention_weights_local,attention_weights_gloabl

class FusionLayer(nn.Module):
    def __init__(self,d_model,num_heads,dff,rate=0.1):
        super(FusionLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = feed_forward_network(d_model, dff)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=rate)

    def forward(self,decoder_embeds,encoder_embeds,encoder_padding_mask,adj_matrix=None,dist_matrix=None):
        """attention_weights_fusion.shape:(batch_size,n_heads,num_atoms+1,num_motif+1)"""
        x_u2m, attention_weights_fusion = self.mha(decoder_embeds, encoder_embeds,encoder_embeds,
                                                             encoder_padding_mask,adj_matrix,dist_matrix)
        x_u2m = self.dropout(x_u2m)
        out = self.layernorm(decoder_embeds + x_u2m)
        x_fusion = self.ffn(out)
        return x_fusion, attention_weights_fusion

class EncoderModel_atom(nn.Module):
    def __init__(self,num_layers,d_model,num_heads,dff,max_length=62,rate=0.1):
        super(EncoderModel_atom,self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        #self.max_length = max_length

        self.embedding = nn.Sequential(nn.Linear(max_length,d_model),nn.ReLU(inplace=False))
        self.global_embedding = nn.Sequential(nn.Linear(d_model,d_model),nn.ReLU(inplace=False))
        self.encoder_layers = nn.ModuleList(EncoderLayer(d_model,num_heads,dff,rate) for _ in range(num_layers))
        self.dropout = nn.Dropout(p=rate)

        ##初始化参数
        self._init_weights()

    def _init_weights(self):
        """使用 Xavier 初始化匹配 TensorFlow 的 glorot_uniform 行为"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'embedding' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def create_padding_mask(self,x):
        """x:(batch_size,seq_len,atom_feats)"""
        atom_padding_mask = torch.eq(torch.sum(x,dim=-1),0).float()
        return atom_padding_mask.unsqueeze(1).unsqueeze(2)

    def forward(self,x,adj_matrix = None,dist_matrix = None,atom_match_matrix = None,sum_atoms = None):
        """x:原子特征，(batch_size,seq_length，62),包含原子特征以及邻居键特征
        atom_match_matrix：原子-子结构匹配矩阵，（batch_size,子结构数，原子数），若子结构包含该原子，则值为1，否则为0
        sum_atoms:（batch_size,子结构数,1），每个元素表示子结构所有的原子数"""
        seq_length = x.size(-1)
        atom_padding_mask = self.create_padding_mask(x)
        if adj_matrix is not None:
            adj_matrix = adj_matrix[:,None,:,:]

        if dist_matrix is not None:
            dist_matrix = dist_matrix[:,None,:,:]

        x = self.embedding(x.float())
        x = self.dropout(x)
        # attention_weights_list_local = []
        # attention_weights_list_global = []
        for i in range(self.num_layers):
            x,_,_ = self.encoder_layers[i](x,atom_padding_mask,adj_matrix,dist_matrix=dist_matrix)
            # attention_weights_list_local.append(attention_weights_local)
            # attention_weights_list_global.append(attention_weights_global)
        x = torch.matmul(atom_match_matrix,x) / sum_atoms #子结构对应的原子级特征，(batch_size,子结构数,512)
        x = self.global_embedding(x)
        return x,None,None,atom_padding_mask

class EncoderModel_motif(nn.Module):
    def __init__(self,num_layers,d_model,num_heads,dff,input_vocab_size,rate=0.1):
        super(EncoderModel_motif, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.encoder_layers = nn.ModuleList(EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers))
        self.dropout = nn.Dropout(p=rate)

        #初始化参数
        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化与 TensorFlow 的 glorot_uniform 对齐"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(self,x):
        """x:(batch_size,seq_length)"""
        motif_padding_mask = torch.eq(x,0).float()
        return motif_padding_mask.unsqueeze(1).unsqueeze(2)

    def forward(self,x,atom_level_features,adj_matrix = None,dist_matrix = None):
        """# x.shape: (batch_size, input_seq_len)掩码原子恢复"""
        seq_length = x.size(1)
        motif_padding_mask = self.create_padding_mask(x)
        if adj_matrix is not None:
            adj_matrix = adj_matrix.unsqueeze(1)

        if dist_matrix is not None:
            dist_matrix = dist_matrix.unsqueeze(1)

        x = self.embedding(x)
        #缩放操作的核心目的是 稳定梯度
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.dropout(x)
        # 将子结构对应的序列索引和对应的原子特征拼接
        x_temp = x[:, 1:, :] + atom_level_features
        x = torch.cat((x[:, 0:1, :], x_temp), dim=1)

        # attention_weights_list_local = []
        # attention_weights_list_global = []
        for i in range(self.num_layers):
            x, _, _ = self.encoder_layers[i](x, motif_padding_mask,adj_matrix,dist_matrix=dist_matrix)
            # attention_weights_list_local.append(attention_weights_local)
            # attention_weights_list_global.append(attention_weights_global)
        return x, None, None, motif_padding_mask

class Fusion_Encoder(nn.Module):
    def __init__(self,num_layers,d_model,num_heads,dff,rate=0.1):
        super(Fusion_Encoder,self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.encoder_layers = nn.ModuleList(FusionLayer(d_model, num_heads, dff, rate) for _ in range(num_layers))
        self.dropout = nn.Dropout(p=rate)

    def forward(self,unimol_feats,motif_feats,motif_padding_mask):
        """unimol_feats: (batch_size, seq_len, d_model)
        motif_feats: (batch_size, motif_len, d_model)"""
        attention_weights_list_fusion = []
        for i in range(self.num_layers):
            unimol_feats, attention_weights_fusion = self.encoder_layers[i](unimol_feats,motif_feats,motif_padding_mask)
            attention_weights_list_fusion.append(attention_weights_fusion)
        return unimol_feats, attention_weights_list_fusion

class MFPK(nn.Module):
    def __init__(self,config):
        super(MFPK,self).__init__()
        self.config = config
        self.dropout = config['dropout']
        self.d_model = config['median']['d_model']
        self.num_layers = config['median']['num_layers']
        self.num_heads = config['median']['num_heads']
        self.dff = config['median']['dff']
        self.input_vocab_size = config['input_vocab_size']
        self.embed_dim = config['embed_dim']
        self.temp = config['temp']
        self.task_num = config['task_num']

        # model_atom_level
        self.encoder_atom = EncoderModel_atom(self.num_layers, self.d_model, self.num_heads, self.dff)
        self.encoder_motif = EncoderModel_motif(self.num_layers, self.d_model, self.num_heads, self.dff,
                                                self.input_vocab_size)

        self.unimol_proj = nn.Linear(self.d_model,self.embed_dim)
        self.motif_proj = nn.Linear(self.d_model,self.embed_dim)

        #fusion encoder
        self.fusion_encoder = Fusion_Encoder(self.num_layers,self.d_model,self.num_heads,self.dff)

        #MLP
        self.mlp = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.d_model,self.embed_dim),
                                 nn.Dropout(self.dropout),nn.ReLU(),
                                 nn.Linear(self.embed_dim,self.task_num))

    def forward(self,unimol_embeds, unimol_embeds_mask,atom_feats,atom_adj_matrix,atom_dist_matrix,atom_match_matrix,sum_atoms,
                motif_num_list,motif_adj_matrix,motif_dist_matrix,args):
        # # unimol model
        #unimol_output = self.unimol.get_repr(smi, return_atomic_reprs=True)
        #print("Unimol_padding",unimol_output['atomic_symbol'])
        # unimol_embeds = torch.tensor(unimol_output['atomic_reprs'], device=args.device)
        # unimol_padding_mask = torch.tensor(unimol_output['atomic_mask'], device=args.device).unsqueeze(1).unsqueeze(2)
        unimol_padding_mask = unimol_embeds_mask.transpose(-2,-1).unsqueeze(1).unsqueeze(2)

        # atom-motif based transformer
        atom_feats, _, _, atom_padding_mask = self.encoder_atom(atom_feats, atom_adj_matrix, atom_dist_matrix,
                                                                atom_match_matrix, sum_atoms)
        motif_feats, _, _, motif_padding_mask = self.encoder_motif(motif_num_list, atom_feats, motif_adj_matrix,
                                                                          motif_dist_matrix)

        #constrastive learning
        motif_feats_global = F.normalize(self.motif_proj(motif_feats[:,0,:]),dim=-1)
        unimol_feats_global = F.normalize(self.unimol_proj(unimol_embeds[:,0,:]),dim=-1)

        sim_u2m = motif_feats_global @ unimol_feats_global.t() / self.temp
        sim_m2u = unimol_feats_global @ motif_feats_global.t() / self.temp

        sim_targets = torch.eye(sim_m2u.size(0),device=args.device)
        loss_m2u = -torch.sum(F.log_softmax(sim_m2u,dim=1) * sim_targets,dim=1).mean()
        loss_u2m = -torch.sum(F.log_softmax(sim_u2m,dim=1) * sim_targets,dim=1).mean()
        constrastive_loss = (loss_m2u + loss_u2m) / 2

        #fusion encoder
        x_fusion, attention_weights_list_fusion = self.fusion_encoder(unimol_embeds,motif_feats,motif_padding_mask)
        x_fusion_global = F.normalize(x_fusion[:,0,:],dim=1)
        prediction = self.mlp(x_fusion_global)

        # 删除中间变量
        del motif_feats, unimol_feats_global, sim_m2u, sim_u2m, x_fusion
        return  attention_weights_list_fusion,prediction,constrastive_loss


    # @torch.no_grad()
    # def concat_all_gather(self,feat):
    #     """
    #     Performs all_gather operation on the provided tensors.
    #     *** Warning ***: torch.distributed.all_gather has no gradient.
    #     """
    #     if not torch.distributed.is_available() or not torch.distributed.is_initialized():
    #         return feat  # 如果 `torch.distributed` 没初始化，直接返回原始数据
    #
    #     tensors_gather = [torch.ones_like(feat) for _ in range(torch.distributed.get_world_size())]
    #     torch.distributed.all_gather(tensors_gather, feat, async_op=False)
    #
    #     output = torch.cat(tensors_gather, dim=0)
    #     return output
    #
    # @torch.no_grad()
    # def _dequeue_and_enqueue(self,motif_feat, unimol_feat):
    #
    #     motif_feats = self.concat_all_gather(motif_feat)
    #     unimol_feats = self.concat_all_gather(unimol_feat)
    #
    #     batch_size = motif_feats.shape[0]
    #
    #     ptr = int(self.queue_ptr)
    #     assert self.queue_size % batch_size == 0  # for simplicity
    #
    #     # replace the keys at ptr (dequeue and enqueue)
    #     self.motif_queue[:, ptr:ptr + batch_size] = motif_feats.T
    #     self.unimol_queue[:, ptr:ptr + batch_size] = unimol_feats.T
    #     ptr = (ptr + batch_size) % self.queue_size  # move pointer
    #
    #     self.queue_ptr[0] = ptr












