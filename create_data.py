import numpy as np

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 400

# 读取蛋白质序列文件
with open('F:/master1/gxy_paper_drug/paper/GPCR/transformerCPI-master/sum/prosum_train_shuf.txt', 'r') as file:
    protein_sequences = file.read().splitlines()

# 转化蛋白质序列为嵌入表示
protein_embeddings = [seq_cat(seq) for seq in protein_sequences]
my_array = np.array(protein_embeddings)
dimensions = my_array.shape
print(dimensions)
# 将 protein_embeddings 保存为 .npy 文件
np.save('F:/master1/gxy_paper_drug/paper/GPCR/transformerCPI-master/GPCR_train2/protein_shuf_trainembeddings.npy', protein_embeddings)

# 将嵌入表示转化为 PyTorch 张量
# protein_embeddings = torch.tensor(protein_embeddings, dtype=torch.float32)
# 现在，protein_embeddings 包含了所有蛋白质序列的嵌入表示，以 PyTorch 张量的形式存储
