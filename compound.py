# -*- encoding: utf-8 -*-
'''
@Time       : 2020/07/15 11:05:03
@Author     : Yize Jiang
@Email      : yize.jiang@galixir.com
@File       : compound.py
@Project    : X-DPI
@Description: 小分子相关特征
'''
import numpy as np
import torch.nn as nn
import torch
from mol2vec.features import (MolSentence, mol2alt_sentence, sentences2vec)
from rdkit import Chem
from const import (ALLEN_NEGATIVITY_TABLE, ATOM_CLASS_TABLE,ELECTRON_AFFINITY_TABLE, ELEMENT_LIST, NUM_ATOM_FEAT, PT)
from utils import one_of_k_encoding, one_of_k_encoding_unk

def atom_features(atom, explicit_H=False, use_chirality=True):
    """Generate atom features including atom symbol(10),degree(7),formal charge,
    radical electrons,hybridization(6),aromatic(1),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2, 'other'
    ]  # 6-dim
    results = (one_of_k_encoding_unk(atom.GetSymbol(), symbol) + one_of_k_encoding(atom.GetDegree(), degree) +
               [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] +
               one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]
               )  # 10+7+2+6+1=26

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])  # 26+5=31
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(atom.GetProp('_CIPCode'),
                                                      ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except Exception:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 31+3 =34
    return results

def atomic_features(atomic_num):
    # symbol = PT.GetElementSymbol(atomic_num)
    # symbol_k = one_of_k_encoding_unk(symbol, ELEMENT_LIST)

    # Period
    outer_electrons = PT.GetNOuterElecs(atomic_num)
    outer_electrons_k = one_of_k_encoding(outer_electrons, list(range(0, 8 + 1)))

    # Default Valence
    default_electrons = PT.GetDefaultValence(atomic_num)  # -1 for transition metals
    default_electrons_k = one_of_k_encoding(default_electrons, list(range(-1, 8 + 1)))

    # Orbitals / Group / ~Row
    orbitals = next(j + 1 for j, val in enumerate([2, 10, 18, 36, 54, 86, 120]) if val >= atomic_num)
    orbitals_k = one_of_k_encoding(orbitals, list(range(0, 7 + 1)))

    # IUPAC Series
    atom_series = ATOM_CLASS_TABLE[atomic_num]
    atom_series_k = one_of_k_encoding(atom_series, list(range(0, 9 + 1)))

    # Centered Electrons
    centered_oec = abs(outer_electrons - 4)

    # Electronegativity & Electron Affinity
    try:
        allen_electronegativity = ALLEN_NEGATIVITY_TABLE[atomic_num]
    except KeyError:
        allen_electronegativity = 0
    try:
        electron_affinity = ELECTRON_AFFINITY_TABLE[atomic_num]
    except KeyError:
        electron_affinity = 0

    # Mass & Radius (van der waals / covalent / bohr 0)
    floats = [
        centered_oec, allen_electronegativity, electron_affinity,
        PT.GetAtomicWeight(atomic_num),
        PT.GetRb0(atomic_num),
        PT.GetRvdw(atomic_num),
        PT.GetRcovalent(atomic_num), outer_electrons, default_electrons, orbitals
    ]
    feature_array = np.array(outer_electrons_k + default_electrons_k + orbitals_k + atom_series_k + floats,
                             dtype=np.float32)
    # Cache in dict for future use
    return feature_array

def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency) + np.eye(adjacency.shape[0])

def get_mol_features(smiles, atom_dim):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        raise RuntimeError("SMILES cannot been parsed!")
    # mol = Chem.AddHs(mol)
    # atom_feat = np.zeros((mol.GetNumAtoms(), NUM_ATOM_FEAT))
    atom_feat = np.zeros((mol.GetNumAtoms(), atom_dim))
    map_dict = dict()
    for atom in mol.GetAtoms():
        # atomic_features(atom.GetAtomicNum())
        # atom_feat[atom.GetIdx(), :] = np.append(atom_features(atom), atomic_features(atom.GetAtomicNum()))
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
        map_dict[atom.GetIdx()] = atom.GetSmarts()
    adj_matrix = adjacent_matrix(mol)
    return atom_feat, adj_matrix, map_dict

def get_mol2vec_features(model, smiles):
    mol = Chem.MolFromSmiles(smiles)
    sen = (MolSentence(mol2alt_sentence(mol, 0)))
    mol_vec = (sentences2vec(sen, model, unseen='UNK'))
    return mol_vec

###图对比学习其中的一个视角
def gen_view1(adj_matrix):
    # 定义 dropout 的概率，通常是一个小的浮点数，例如 0.2 表示 20% 的边将被设置为零
    dropout_prob = 0.2
    # 创建一个 Dropout 层，其中 p 参数表示 dropout 的概率
    dropout_layer = nn.Dropout(p=dropout_prob)
    # 对邻接矩阵应用 dropout
    adj_matrix_with_dropout = dropout_layer(torch.from_numpy(adj_matrix))
    return adj_matrix_with_dropout

def gen_view2(self,embs,re_rate = 0.2):

    embs_mean = embs.mean(0).unsqueeze(0)
    embs_mean = torch.repeat_interleave(embs_mean,repeats=embs.shape[0],dim= 0)
    mask = torch.rand_like(embs)
    mask = torch.where(mask < re_rate,1,0)
    embs_v2 = torch.where(mask == 1,embs_mean,embs)
    return embs_v2