import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from core_LR import Predictor
from data import DPIDataset
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, random_split
from sklearn.linear_model import LogisticRegression
#import neptune
#device = torch.device('cuda')

def run(args: argparse.Namespace):
    batch_size = args.batchsize
    epochs = args.epochs
    lr = args.lr
    dropout = args.dropout
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    n_filters = args.n_filters
    embed_dim = args.embed_dim
    num_features_xt = args.num_features_xt
    output_dim = args.output_dim

    proesmtr = args.train_proteinesm_path
    protein_trainembeddings = args.train_proteincnn_path
    drmol = args.train_drugmol_path
    train_druggcn = args.train_druggcn_path
    train_label = args.train_label_path

    proesmte = args.test_proteinesm_path
    protein_testembeddings = args.test_proteincnn_path
    demol = args.test_drugmol_path
    test_druggcn = args.test_druggcn_path
    test_label = args.test_label_path

    # 数据集划分
    train_set = DPIDataset(proesmtr, protein_trainembeddings, drmol, train_druggcn, train_label, args)
    val_size = int(0.2 * len(train_set))
    train_size = len(train_set) - val_size
    # 使用 random_split 分割数据集
    train_dataset, val_dataset = random_split(train_set, [train_size, val_size])
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_set.collate_fn,
                              drop_last=True)
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=train_set.collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_set.collate_fn,
                            drop_last=True)
    test_set = DPIDataset(proesmte, protein_testembeddings, demol, test_druggcn, test_label, args)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=test_set.collate_fn,
                             drop_last=True)
    # 定义模型Predictor
    model = Predictor(input_dim, hidden_dim, dropout, n_filters, embed_dim, num_features_xt, output_dim, args)
    #model.to(device)
    # 打印模型的权重和偏差
    '''
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter name: {name}")
            print(f"Parameter value: {param.data}")
            print("=" * 20)'''
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    train_epoch_size = len(train_label)
    test_epoch_size = len(test_label)
    Tmax_auc = 0.0  # 用于存储所有epoch中的最大的test_AUC值
    Tmax_prc = 0.0  # 用于存储所有epoch中的最大的test_PRC值

    '''
    run = neptune.init_run(
        project="gxy14550221/MLP",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4YmNjNTkwNy1mNjc0LTQwMjYtYmQ3Yi0zOGI2MjJmZDI5M2EifQ==", )
    run["sys/tags"].add(['test1', 'test2'])'''

    print('--- MLP model --- ')
    for epoch in range(epochs):
        # ---------------train------------
        model.train()

        pred_prob = []
        total_predicted = torch.Tensor()
        total_labels = torch.Tensor()
        total_pred_prob = torch.Tensor()
        total_preds = torch.Tensor()
        total_train_loss = 0.0  # 添加此行以跟踪总的训练损失

        for batch_idx, data in enumerate(train_loader):
            ###对每一个batch_size进行特征提取以及MLP的计算
            #data = {key: value.to(device) for key, value in data.items()}
            data = {key: value for key, value in data.items()}
            #label = data['LABEL'].float().to(device)
            label = data['LABEL'].float()
            esm_embedding = data['PROTEIN_ESM_EMBEDDING']
            #esm_embedding = esm_embedding.to(device)
            esm_embedding = esm_embedding
            cnn_embedding = data['PROTEIN_CNN_EMBEDDING']
            #cnn_embedding = cnn_embedding.to(device)
            cnn_embedding = cnn_embedding
            mol2vec_embedding = data['COMPOUND_MOL_EMBEDDING']
            #mol2vec_embedding = mol2vec_embedding.to(device)
            mol2vec_embedding = mol2vec_embedding

            pred_fea = model(data, esm_embedding, cnn_embedding, mol2vec_embedding)
            predicted = (pred_fea > 0.5).int()
            ###创建逻辑回归分类器
            lr_classifier = LogisticRegression()
            lr_classifier.fit(predicted, label)

        # ---------------test------------
        max_test_acc = 0.0
        max_test_precision = 0.0
        max_test_recall = 0.0
        max_test_f1 = 0.0
        max_test_mcc = 0.0
        max_test_auc = 0.0
        max_test_prc = 0.0
        total_pred_test = torch.Tensor()
        total_predicted_test = torch.Tensor()
        total_label = torch.Tensor()
        total_pred_prob_test = torch.Tensor()

        with torch.no_grad():
            for data in test_loader:
                '''
                data = {key: value.to(device) for key, value in data.items()}
                label_test = data['LABEL'].float().to(device)
                esm_embedding = data['PROTEIN_ESM_EMBEDDING']
                esm_embedding = esm_embedding.to(device)
                cnn_embedding = data['PROTEIN_CNN_EMBEDDING']
                cnn_embedding = cnn_embedding.to(device)
                mol2vec_embedding = data['COMPOUND_MOL_EMBEDDING']
                mol2vec_embedding = mol2vec_embedding.to(device)'''
                data = {key: value for key, value in data.items()}
                label_test = data['LABEL'].float()
                esm_embedding = data['PROTEIN_ESM_EMBEDDING']
                esm_embedding = esm_embedding
                cnn_embedding = data['PROTEIN_CNN_EMBEDDING']
                cnn_embedding = cnn_embedding
                mol2vec_embedding = data['COMPOUND_MOL_EMBEDDING']
                mol2vec_embedding = mol2vec_embedding
                ###提取的特征
                pred_test_fea = model(data, esm_embedding, cnn_embedding, mol2vec_embedding)
                predicted_test = lr_classifier.predict(pred_test_fea)
                predicted_test = torch.tensor(predicted_test)

                ###每个样本属于每个类别的概率分布
                pred_prob_test = F.softmax(predicted_test, dim=-1)

                total_predicted_test = torch.cat((total_predicted_test, predicted_test.cpu()), 0)
                total_label = torch.cat((total_label, label_test.view(-1, 1).cpu()), 0)
                total_pred_test = torch.cat((total_pred_test, predicted_test.cpu()), 0)
                total_pred_prob_test = torch.cat((total_pred_prob_test, pred_prob_test.cpu()), 0)

            acc = accuracy_score(total_label, total_predicted_test)
            precision = precision_score(total_label, total_predicted_test)
            recall = recall_score(total_label, total_predicted_test)
            f1 = f1_score(total_label, total_predicted_test)
            mcc = matthews_corrcoef(total_label, total_predicted_test)
            roc_auc = roc_auc_score(total_label, total_pred_prob_test)
            prc = average_precision_score(total_label, total_pred_prob_test)

            # Update maximum PRC and AUC values
            max_test_acc = max(max_test_acc, acc)
            max_test_auc = max(max_test_auc, roc_auc)
            max_test_prc = max(max_test_prc, prc)
            if acc > max_test_acc:
                max_test_acc = acc
                max_test_precision = precision
                max_test_recall = recall
                max_test_f1 = f1
                max_test_mcc = mcc

            print('# [{}/{}] testing {:.1%}, acc={:.5f}, precision={:.5f}, recall={:.5f}, f1={:.5f}, mcc={:.5f}, roc_auc={:.5f}, prc={:.5f}\n'.format(
                    epoch + 1, epochs, (epoch + 1) / epochs, acc, precision, recall, f1, mcc, roc_auc, prc), end='\r')
        Tmax_auc = max(Tmax_auc, max_test_auc)
        Tmax_prc = max(Tmax_prc, max_test_prc)
    # 在所有epoch结束后打印最大AUC和PRC
    print('Best test_acc: {:.5f}'.format(max_test_acc))
    print('Best test_precision: {:.5f}'.format(max_test_precision))
    print('Best test_recall: {:.5f}'.format(max_test_recall))
    print('Best test_f1: {:.5f}'.format(max_test_f1))
    print('Best test_mcc: {:.5f}'.format(max_test_mcc))
    print('Finally!Max test_AUC over all epochs: {:.5f}'.format(Tmax_auc))
    print('Finally!Max test_PRC over all epochs: {:.5f}'.format(Tmax_prc))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs to train.')
    parser.add_argument('--batchsize', type=int, default=100, help='Number of batch_size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')  # 1e-4
    parser.add_argument('--hidden-dim', type=int, default=300, help='hidden units of FC layers (default: 256)')
    parser.add_argument('--input_dim', type=int, default=576, help='input_dim.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')  # 0.1

    parser.add_argument('--n_filters', type=int, default=64, help='n_filters')
    parser.add_argument('--embed_dim', type=int, default=128, help='embed_dim')
    parser.add_argument('--num_features_xt', type=int, default=25, help='num_features_xt')
    parser.add_argument('--output_dim', type=int, default=128, help='output_dim')
    parser.add_argument('--max_compound_len', type=int, default=300, help='output_dim')

    parser.add_argument('--seed', type=int, default=2020, help='Random Seed')
    parser.add_argument('--gpus', type=str, default='0', help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of Subprocesses for Data Loading')
    parser.add_argument('--early_stop_round', type=int, default=5, help='Early Stopping Round in Validation')

    parser.add_argument('--decoder_layers', type=int, default=3, help='Number of Layers for Decoder')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of Heads for Attention')
    parser.add_argument('--gnn_layers', type=int, default=3, help='Layers of GNN')
    parser.add_argument('--compound_gnn_dim', type=int, default=64, help='Hidden Dimension for Attention')
    parser.add_argument('--hid_dim', type=int, default=64, help='Hidden Dimension for Attention')
    parser.add_argument('--pf_dim', type=int, default=256, help='Hidden Dimension for Positional Feed Forward')

    parser.add_argument('--mol2vec_embedding_dim', type=int, default=300, help='Dimension for Mol2vec Embedding')
    parser.add_argument('--valid_test', type=bool, default=False, help='Testing for validation data')
    parser.add_argument('--atom_dim', type=int, default=34, help='Dimension for Atom')
    parser.add_argument('--embedding_dim', type=int, default=768, help='Dimension for Embedding')

    parser.add_argument('--cnn_kernel_size', type=int, default=8, help='CNN Conv Layers')

    parser.add_argument('--train-proteinesm-path',
                        default='F:/master1/gxy_paper_drug/paper/GPCR/transformerCPI-master/GPCR_train2/pro_esmtr.npy',
                        help='training-dataset-path or not')
    parser.add_argument('--train-proteincnn-path',
                        default='F:/master1/gxy_paper_drug/paper/GPCR/transformerCPI-master/GPCR_train2/protein_trainembeddings400.npy',
                        help='training-dataset-path or not')
    parser.add_argument('--train-drugmol-path',
                        default='F:/master1/gxy_paper_drug/paper/GPCR/transformerCPI-master/GPCR_train2/dr_moltr.npy',
                        help='training-dataset-path or not')
    parser.add_argument('--train-druggcn-path',
                        default='F:/master1/gxy_paper_drug/paper/GPCR/transformerCPI-master/GPCR_train2/drug_train.csv',
                        help='training-dataset-path or not')
    parser.add_argument('--train-label-path',
                        default='F:/master1/gxy_paper_drug/paper/GPCR/transformerCPI-master/GPCR_train2/label_train.npy',
                        help='testing-dataset-path or not')

    parser.add_argument('--test-proteinesm-path',
                        default='F:/master1/gxy_paper_drug/paper/GPCR/transformerCPI-master/GPCR_test2/pro_esmte_shut.npy',
                        help='testing-dataset-path or not')
    parser.add_argument('--test-proteincnn-path',
                        default='F:/master1/gxy_paper_drug/paper/GPCR/transformerCPI-master/GPCR_test2/protein_testembeddings400.npy',
                        help='testing-dataset-path or not')
    parser.add_argument('--test-drugmol-path',
                        default='F:/master1/gxy_paper_drug/paper/GPCR/transformerCPI-master/GPCR_test2/temol_shut.npy',
                        help='training-dataset-path or not')
    parser.add_argument('--test-druggcn-path',
                        default='F:/master1/gxy_paper_drug/paper/GPCR/transformerCPI-master/GPCR_test2/drug_test_shut.csv',
                        help='testing-dataset-path or not')
    parser.add_argument('--test-label-path',
                        default='F:/master1/gxy_paper_drug/paper/GPCR/transformerCPI-master/GPCR_test2/label_test_shut.npy',
                        help='testing-dataset-path or not')

    parser.add_argument('--objective', type=str, default='classification',
                        help='Objective (classification / regression)')
    return parser.parse_args()

if __name__ == '__main__':
    params = parse_args()
    print(params)
    torch.cuda.manual_seed_all(params.seed)
    run(params)