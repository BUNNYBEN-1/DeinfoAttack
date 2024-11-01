import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
import numpy as np


class FALayer(nn.Module):
    def __init__(self, g, in_dim, dropout):
        super(FALayer, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(dropout)

        # 定义一个线性层，用于计算门控权重
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)  # 使用Xavier初始化方法初始化线性层的权重

    def edge_applying(self, edges):
        # 对每一条边应用操作
        h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()
        e = g * edges.dst['d'] * edges.src['d']
        e = self.dropout(e)
        return {'e': e, 'm': g}

    def forward(self, h):
        # 设置节点特征
        self.g.ndata['h'] = h

        # 对每一条边应用edge_applying操作
        self.g.apply_edges(self.edge_applying)

        # 将边的信息进行汇总
        self.g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))

        return self.g.ndata['z']


class FAGCN(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, dropout, eps, layer_num=2):
        super(FAGCN, self).__init__()
        # 初始化参数，g：图G，in_dim：输入特征维度
        self.g = g
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        # 创建多层FALayer，并将它们存储在nn.ModuleList中
        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(self.g, hidden_dim, dropout))

        # 定义线性层t1和t2
        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.t2 = nn.Linear(hidden_dim, out_dim)
        
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        # 使用xavier_normal_初始化线性层的权重
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        # 对输入特征进行dropout操作
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # 应用线性层t1和激活函数relu
        h = torch.relu(self.t1(h))
        
        # 再次对特征进行dropout操作
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # 保存原始特征
        raw = h

        # 循环遍历FALayer，并应用到特征上
        for i in range(self.layer_num):
            h = self.layers[i](h)
            h = self.eps * raw + h
        
        # 应用线性层t2和softmax激活函数
        h = self.t2(h)
        return F.log_softmax(h, 1)

