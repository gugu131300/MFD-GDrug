import torch
import torch.nn as nn

# GCN based model
class GCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128, num_features_xt=25, output_dim=128, dropout=0.2):

        super(GCNNet, self).__init__()

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=500, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

    def forward(self, sequence):

        # 1d conv layers
        embedded_xt = self.embedding_xt(sequence)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)
        #print("xt",xt)
        return xt