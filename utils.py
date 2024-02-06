# -*- encoding: utf-8 -*-
'''
@Time       : 2020/07/18 13:53:49
@Author     : Yize Jiang
@Email      : yize.jiang@galixir.com
@File       : gnn.py
@Project    : X-DPI
@Description: 图神经网络
'''

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class MultiGCN(nn.Module):
    def __init__(self, args, in_dim, out_dim):
        super(MultiGCN, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim, bias=True)
        self.fc2 = nn.Linear(out_dim, out_dim, bias=True)
        self.fc3 = nn.Linear(out_dim, out_dim, bias=True)

    def forward(self, inputs, adj):
        outputs1 = torch.bmm(adj, self.fc1(inputs))
        outputs2 = torch.bmm(adj, self.fc2(outputs1))
        outputs3 = torch.bmm(adj, self.fc3(outputs2))
        return (outputs3+outputs2+outputs1)/3

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

class LinkAttention_pro(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(LinkAttention_pro, self).__init__()
        self.query = nn.Linear(input_dim, n_heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, masks):
        query = self.query(x).transpose(1, 2)
        value = x
        value = value.transpose(1, 2)  # 交换第二个维度（1）和第三个维度（100）
        minus_inf = -9e15 * torch.ones_like(query)
        e = torch.where(masks > 0.5, query, minus_inf)  # (B,heads,seq_len)
        a = self.softmax(e)

        out = torch.matmul(a, value)
        out = torch.sum(out, dim=1).squeeze()
        return out, a

class LinkAttention_drug(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(LinkAttention_drug, self).__init__()
        self.query = nn.Linear(input_dim, n_heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, masks):
        query = self.query(x).transpose(1, 2)
        value = x
        minus_inf = -9e15 * torch.ones_like(query)
        e = torch.where(masks > 0.5, query, minus_inf)  # (B,heads,seq_len)
        a = self.softmax(e)
        out = torch.matmul(a, value)
        out = torch.sum(out, dim=1).squeeze()
        return out, a