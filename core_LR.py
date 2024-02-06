import torch
import torch.nn as nn
from utils import MultiGCN
import torch.nn.functional as F

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate, n_filters, embed_dim, num_features_xt, output_dim, args):
        super(Predictor, self).__init__()

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=400, out_channels=n_filters, kernel_size=7)  #######这里有需要修改的参数
        self.fc1_xt = nn.Linear(32*244, output_dim)

        # drug
        self.atom_dim = args.atom_dim
        self.embedding_dim = args.embedding_dim
        self.mol2vec_embedding_dim = args.mol2vec_embedding_dim
        self.compound_gnn_dim = args.compound_gnn_dim
        decoder_hid_dim = args.hid_dim * 4  # for Classification
        self.compound_gcn = MultiGCN(args, self.atom_dim, args.compound_gnn_dim)
        out_features_fc3 = self.compound_gcn.fc3.out_features
        self.mol2vec_fc = nn.Linear(self.mol2vec_embedding_dim, self.mol2vec_embedding_dim)
        self.mol_concat_fc = nn.Linear(self.mol2vec_embedding_dim + out_features_fc3 , decoder_hid_dim)
        self.mol_concat_ln = nn.LayerNorm(decoder_hid_dim)

        # combined layers
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.out_fc1 = nn.Linear(input_dim, hidden_dim)
        self.out_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out_fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, batch, esm_embedding, cnn_embedding, mol2vec_embedding):

        # protein cnn
        # 1d conv layers
        cnn_embedding = cnn_embedding.to(torch.int)
        embedded_xt = self.embedding_xt(cnn_embedding)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 244)
        xt = self.fc1_xt(xt)
        xt = F.relu(xt)

        # comb_protein
        combined_pro = torch.cat((xt, esm_embedding), 1)  # 在列方向上叠加

        # drug
        compound_gcn = self.compound_gcn(batch["COMPOUND_NODE_FEAT"], batch["COMPOUND_ADJ"])
        ###当前compound的最大值
        mean_tensor = torch.max(compound_gcn, dim=1).values
        mol2vec_embedding = self.mol2vec_fc(mol2vec_embedding)

        # comb_protein
        compound = torch.cat([mol2vec_embedding, mean_tensor], dim=-1)
        compound = self.mol_concat_fc(compound)
        compound = self.mol_concat_ln(compound)

        # comb
        combined_fea = torch.cat((compound, esm_embedding), 1)  # 在列方向上叠加 #########  # 在列方向上叠加 #########

        return combined_fea