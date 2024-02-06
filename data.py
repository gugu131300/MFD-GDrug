import pandas as pd
import torch
from torch.utils.data import DataLoader
from compound import get_mol_features, get_mol2vec_features, gen_view1
import numpy as np
from gensim.models import word2vec

if torch.cuda.is_available():
    device = torch.device('cuda')

class DPIDataset():
    """CPI数据集
    Args:
        file_path: 数据文件路径（包含化合物SMILES、蛋白质氨基酸序列、标签）
    """

    def __init__(self, protein_pretrained, protein_cnn, drug_pretrained, protein_gcn, label, args):
        self.protein_pretrained = np.load(protein_pretrained)
        self.protein_cnn = np.load(protein_cnn)
        self.drug_pretrained = np.load(drug_pretrained)
        self.raw_data = pd.read_csv(protein_gcn)
        self.smiles_values = self.raw_data['COMPOUND_SMILES'].values
        self.label = np.load(label)
        self.atom_dim = args.atom_dim
        self.mol2vec_model = word2vec.Word2Vec.load("F:/master1/gxy_paper_drug/paper/GPCR/transformerCPI-master/GPCR_train2/model_300dim.pkl")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        ###drug的smiles_gcn
        protein_pretrained = self.protein_pretrained[idx]
        protein_cnn = self.protein_cnn[idx]
        drug_pretrained = self.drug_pretrained[idx]
        smiles = self.smiles_values[idx]
        compound_node_features, compound_adj_matrix, _ = get_mol_features(smiles, self.atom_dim)
        ###对比学习中v1视角下的graph
        compound_adj_matrix_v1 = gen_view1(compound_adj_matrix)

        # compound_word_embedding = self.embedding_data[idx]  # 使用embedding数据
        label = self.label[idx]
        protein_len = protein_cnn.shape[0]
        compound_word_embedding = get_mol2vec_features(self.mol2vec_model, smiles)
        compound_word_embedding = np.array(compound_word_embedding)

        return {
            'PROTEIN_ESM_EMBEDDING': protein_pretrained,
            'PROTEIN_CNN_EMBEDDING': protein_cnn,
            'COMPOUND_MOL_EMBEDDING': drug_pretrained,
            'COMPOUND_NODE_FEAT': compound_node_features,
            'COMPOUND_ADJ': compound_adj_matrix,
            'LABEL': label,
            'PROTEIN_LEN': protein_len,
            'COMPOUND_WORD_EMBEDDING': compound_word_embedding,
            'COMPOUND_ADJ_V1': compound_adj_matrix_v1
        }

    def collate_fn(self, batch):
        """自定义数据合并方法，将数据集中的数据通过Padding构造成相同Size
        Args:
            batch: 原始数据列表
        Returns:
            batch: 经过Padding等处理后的PyTorch Tensor字典
        """
        batch_size = len(batch)

        protein_pretrained, protein_cnn, drug_pretrained, compound_node_features, compound_adj_matrix, label, protein_len, compound_word_embedding, compound_adj_matrix_v1 = \
            zip(*[(item['PROTEIN_ESM_EMBEDDING'], item['PROTEIN_CNN_EMBEDDING'], item['COMPOUND_MOL_EMBEDDING'],
                   item['COMPOUND_NODE_FEAT'], item['COMPOUND_ADJ'], item['LABEL'], item['PROTEIN_LEN'], item['COMPOUND_WORD_EMBEDDING'], item['COMPOUND_ADJ_V1']) for item in
                  batch])

        ###drug graph
        compound_node_nums = [item['COMPOUND_NODE_FEAT'].shape[0] for item in batch]
        max_compound_len = 300
        compound_node_features = torch.zeros((batch_size, max_compound_len, batch[0]['COMPOUND_NODE_FEAT'].shape[1]))
        compound_adj_matrix = torch.zeros((batch_size, max_compound_len, max_compound_len))
        compound_word_embedding = torch.zeros((batch_size, max_compound_len, batch[0]['COMPOUND_WORD_EMBEDDING'].shape[1]))
        compound_adj_matrix_v1 = torch.zeros((batch_size, max_compound_len, max_compound_len))

        labels = list()

        for i, item in enumerate(batch):
            v = item['COMPOUND_NODE_FEAT']
            compound_node_features[i, :v.shape[0], :] = torch.FloatTensor(v)
            # print("(compound_node_features[i, :v.shape[0], :]).shape", (compound_node_features[i, :v.shape[0], :]).shape)
            v = item['COMPOUND_ADJ']
            compound_adj_matrix[i, :v.shape[0], :v.shape[0]] = torch.FloatTensor(v)
            v = item['COMPOUND_ADJ_V1']
            v = v.clone().detach().requires_grad_(True)
            compound_adj_matrix_v1[i, :v.shape[0], :v.shape[0]] = v
            v = item['COMPOUND_WORD_EMBEDDING']
            compound_word_embedding[i, :v.shape[0], :] = torch.FloatTensor(v)
            # print("(compound_adj_matrix[i, :v.shape[0], :v.shape[0]]).shape", (compound_adj_matrix[i, :v.shape[0], :v.shape[0]]).shape)
            compound_node_nums = torch.LongTensor(compound_node_nums)
            labels.append(item['LABEL'])
        '''
        compound_node_features.to(device)
        compound_adj_matrix.to(device)
        compound_adj_matrix_v1.to(device)
        compound_node_nums.to(device)
        protein_cnn = np.array(protein_cnn)
        protein_cnn = torch.FloatTensor(protein_cnn).to(device)
        protein_pretrained = np.array(protein_pretrained)
        protein_pretrained = torch.FloatTensor(protein_pretrained).to(device)
        drug_pretrained = np.array(drug_pretrained)
        drug_pretrained = torch.FloatTensor(drug_pretrained).to(device)
        labels = np.array(labels)
        labels = torch.tensor(labels).to(device)'''
        protein_cnn = np.array(protein_cnn)
        protein_cnn = torch.FloatTensor(protein_cnn)
        protein_pretrained = np.array(protein_pretrained)
        protein_pretrained = torch.FloatTensor(protein_pretrained)
        drug_pretrained = np.array(drug_pretrained)
        drug_pretrained = torch.FloatTensor(drug_pretrained)
        labels = np.array(labels)
        labels = torch.tensor(labels)
        compound_word_embedding = np.array(compound_word_embedding)
        compound_word_embedding = torch.FloatTensor(compound_word_embedding)

        return {
            'COMPOUND_NODE_FEAT': compound_node_features,
            'COMPOUND_ADJ': compound_adj_matrix,
            'COMPOUND_ADJ_V1': compound_adj_matrix_v1,
            'COMPOUND_NODE_NUM': compound_node_nums,
            'PROTEIN_CNN_EMBEDDING': protein_cnn,
            'PROTEIN_ESM_EMBEDDING': protein_pretrained,
            'COMPOUND_MOL_EMBEDDING': drug_pretrained,
            'COMPOUND_WORD_EMBEDDING': compound_word_embedding,
            'LABEL': labels,
        }

class Args:
    def __init__(self):
        self.atom_dim = 34  # 设置atom_dim的值

if __name__ == '__main__':
    args = Args()  # 创建Args对象并设置atom_dim的值
    torch.multiprocessing.set_start_method('spawn')

    data_set = DPIDataset(protein_pretrained='/mnt/sda/guxingyue/DPI/data/pro_esmtr.npy',
                          protein_cnn='/mnt/sda/guxingyue/DPI/data/protein_trainembeddings.npy',
                          drug_pretrained='/mnt/sda/guxingyue/DPI/data/dr_moltr.npy',
                          protein_gcn='/mnt/sda/guxingyue/DPI/data/drug_train.csv',
                          label='/mnt/sda/guxingyue/DPI/data/label_train.npy',
                          args=args)

    item = data_set[1]
    print('Item:')
    print('Protein_pretrained:', item['PROTEIN_ESM_EMBEDDING'].shape)
    print('Protein_cnn:', item['PROTEIN_CNN_EMBEDDING'].shape)
    print('Drug_pretrained:', item['COMPOUND_MOL_EMBEDDING'].shape)
    print('Compound_node_features:', item['COMPOUND_NODE_FEAT'].shape)
    print('Compound_adj_matrix:', item['COMPOUND_ADJ'].shape)
    print('Label:', item['LABEL'])
    print('protein_len', item['PROTEIN_LEN'])

    data_loader = DataLoader(
        data_set,
        batch_size=32,
        collate_fn=data_set.collate_fn(data_set),
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    print('')
    print('Batch:')
    for batch in data_loader:
        # print("batch",batch)
        print('Protein_pretrained:', batch['PROTEIN_ESM_EMBEDDING'].shape)
        print('Protein_cnn:', batch['PROTEIN_CNN_EMBEDDING'].shape)
        print('Drug_pretrained:', batch['COMPOUND_MOL_EMBEDDING'].shape)
        print('Compound_node_features:', batch['COMPOUND_NODE_FEAT'].shape)
        print('Compound_adj_matrix:', batch['COMPOUND_ADJ'].shape)
        print('Label:', batch['LABEL'].shape)
        break