import argparse
import os
import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from core import Predictor
from data import DPIDataset
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score, precision_score, recall_score
from torch.utils.data import DataLoader, random_split
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_set.collate_fn,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_set.collate_fn,drop_last=True)
    test_set = DPIDataset(proesmte, protein_testembeddings, demol, test_druggcn, test_label, args)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=test_set.collate_fn,drop_last=True)
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

    '''
    run = neptune.init_run(
        project="gxy14550221/MLP",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4YmNjNTkwNy1mNjc0LTQwMjYtYmQ3Yi0zOGI2MjJmZDI5M2EifQ==", )
    run["sys/tags"].add(['test1', 'test2'])'''

    print('--- MLP model --- ')
    max_test_epoch = -1
    max_test_prc = 0.0
    max_test_auc = 0.0
    max_test_prec = 0.0
    max_test_recall = 0.0
    total_test_loss = 0.0
    for epoch in range(epochs):
        # ---------------train------------
        model.train()

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

            pred, combined_fea = model(data, esm_embedding, cnn_embedding, mol2vec_embedding)
            predicted = (pred > 0.5).int()

            affinity = label.squeeze()
            loss = criterion(pred, affinity)
            #loss = loss.to(device)

            ##清除之前的梯度
            optim.zero_grad()
            loss.backward()
            optim.step()

            ###每个样本属于每个类别的概率分布
            pred_prob = F.softmax(pred, dim=-1)

            total_train_loss += loss.item()  # 更新总的训练损失
            total_predicted = torch.cat((total_predicted, predicted.cpu()), 0)
            total_labels = torch.cat((total_labels, label.view(-1, 1).cpu()), 0)
            total_pred_prob = torch.cat((total_pred_prob, pred_prob.cpu()), 0)
            total_preds = torch.cat((total_preds, pred.cpu()), 0)

        avg_train_loss = (total_train_loss / train_epoch_size) * batch_size  # 计算平均训练损失
        total_labels = total_labels.detach().cpu().numpy()
        total_predicted = total_predicted.detach().cpu().numpy()
        total_pred_prob = total_pred_prob.detach().cpu().numpy()

        acc = accuracy_score(total_labels, total_predicted)
        mcc = matthews_corrcoef(total_labels, total_predicted)
        roc_auc = roc_auc_score(total_labels, total_pred_prob)

        #run['train_loss'].append(avg_train_loss)
        print('# [{}/{}] training {:.1%}, acc={:.5f}, mcc={:.5f}, auc={:.5f}\n'.format(epoch + 1, epochs,
                                                                                       (epoch + 1) / epochs, acc, mcc,
                                                                                       roc_auc,
                                                                                       end='\r'))
        # ---------------test------------
        model.eval()
        total_pred_test = torch.Tensor()
        total_predicted_test = torch.Tensor()
        total_label_test = torch.Tensor()
        total_pred_prob_test = torch.Tensor()
        total_fea_test = torch.Tensor()

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
                pred_test, combined_fea_test = model(data, esm_embedding, cnn_embedding, mol2vec_embedding)

                predicted_test = (pred_test > 0.5).int()

                ###每个样本属于每个类别的概率分布
                pred_prob_test = F.softmax(pred_test, dim=-1)

                affinity = label_test.squeeze()
                loss = criterion(pred_test, affinity)
                #loss = loss.to(device)
                total_test_loss += loss.item()
                total_predicted_test = torch.cat((total_predicted_test, predicted_test.cpu()), 0)
                total_label_test = torch.cat((total_label_test, label_test.view(-1, 1).cpu()), 0)
                total_pred_test = torch.cat((total_pred_test, pred_test.cpu()), 0)
                total_pred_prob_test = torch.cat((total_pred_prob_test, pred_prob_test.cpu()), 0)
                total_fea_test = torch.cat((total_fea_test, combined_fea_test.cpu()), 0)

            if epoch % 1 == 0:
                np.savetxt("result_eps/total_combined_fea_test_{}.txt".format(epoch + 1), total_fea_test.detach().cpu())
                np.savetxt("result_eps/total_label_test_{}.txt".format(epoch + 1), total_labels)

                x_data = total_fea_test.detach()  # 需要可视化的数据
                y_data = total_label_test# 可视化的数据对应的label，label可以是true label，或者是分类or聚类后对应的label
                X = np.array(x_data)
                y = np.array(y_data).astype(int)
                '''t-SNE'''
                tsne = manifold.TSNE(n_components=2, init='random', random_state=1, learning_rate=200.0)  # n_components=2降维为2维并且可视化
                X_tsne = tsne.fit_transform(X)
                '''空间可视化'''
                x_min, x_max = X_tsne.min(0), X_tsne.max(0)
                X_norm = X_tsne

                plt.figure(figsize=(15, 15))
                plt.xlim([x_min[0] - 5, x_max[0] + 5])
                plt.ylim([x_min[1] - 5, x_max[1] + 5])
                # for i in range(X_norm.shape[0]):
                # plt.text(X_norm[i, 0], X_norm[i, 1], str('.'), color=c[y[i]], fontdict={'weight': 'bold', 'size': 40})
                # plt.show()
                save_dir = "result_eps"
                np.savetxt(os.path.join(save_dir, "Embeds_{}_epoch_{}.txt".format("DPI", epoch + 1)), X_norm)
                np.savetxt(os.path.join(save_dir, "labels_{}_epoch_{}.txt".format("DPI", epoch + 1)), y)

                # 通过不同的颜色表示不同的标签值
                colors = ['red', 'blue']  # 这里根据您的标签值设置颜色
                #plt.figure(figsize=(8, 8))
                for i in range(len(X_norm)):
                    plt.scatter(X_norm[i, 0], X_norm[i, 1], color=colors[int(y[i])])
                plt.xlabel('X-axis')
                plt.ylabel('Y-axis')
                plt.title('Scatter Plot')
                plt.savefig(os.path.join(save_dir, "tsen_Random_features_{}_epoch_{}.eps".format("DPI", epoch + 1)))
                #plt.show()

            acc = accuracy_score(total_label_test, total_predicted_test)
            mcc = matthews_corrcoef(total_label_test, total_predicted_test)
            roc_auc = roc_auc_score(total_label_test, total_pred_prob_test)
            prc = average_precision_score(total_label_test, total_pred_prob_test)
            prec = precision_score(total_label_test, total_predicted_test)
            recall = recall_score(total_label_test, total_predicted_test)

            if roc_auc > max_test_auc:
                max_test_epoch = epoch + 1
                max_test_recall = recall
                max_test_prec = prec
                max_test_auc = roc_auc
                max_test_prc = prc

            #run['test_loss'].append(avg_test_loss)
            print('# [{}/{}] testing {:.1%}, roc_auc={:.5f}, prc={:.5f}, acc={:.5f}, mcc={:.5f}, prec={:.5f}, recall={:.5f}\n'.format(
                epoch + 1, epochs, (epoch + 1) / epochs, roc_auc, prc, acc, mcc, prec, recall), end='\r')
            print('auc improved at epoch ', max_test_epoch, '; best_auc, best_prc, best_precision, best_recall:', max_test_auc,
                  max_test_prc, max_test_prec, max_test_recall)

        # ---------------valid------------
        model.eval()
        val_loss = 0.0
        val_total_labels = torch.Tensor()
        val_total_predicted = torch.Tensor()
        val_total_pred_prob = torch.Tensor()
        with torch.no_grad():
            for data in val_loader:  # 使用验证集数据加载器
                '''
                data = {key: value.to(device) for key, value in data.items()}
                label_val = data['LABEL'].float().to(device)
                esm_embedding = data['PROTEIN_ESM_EMBEDDING']
                esm_embedding = esm_embedding.to(device)
                cnn_embedding = data['PROTEIN_CNN_EMBEDDING']
                cnn_embedding = cnn_embedding.to(device)
                mol2vec_embedding = data['COMPOUND_MOL_EMBEDDING']
                mol2vec_embedding = mol2vec_embedding.to(device)'''
                data = {key: value for key, value in data.items()}
                label_val = data['LABEL'].float()
                esm_embedding = data['PROTEIN_ESM_EMBEDDING']
                esm_embedding = esm_embedding
                cnn_embedding = data['PROTEIN_CNN_EMBEDDING']
                cnn_embedding = cnn_embedding
                mol2vec_embedding = data['COMPOUND_MOL_EMBEDDING']
                mol2vec_embedding = mol2vec_embedding

                pred_val, combined_fea_valid = model(data, esm_embedding, cnn_embedding, mol2vec_embedding)

                # 计算验证集的损失
                val_loss += criterion(pred_val, label_val.squeeze()).item()

                # 记录验证集的预测结果
                val_total_labels = torch.cat((val_total_labels, label_val.view(-1, 1).cpu()), 0)
                val_total_predicted = torch.cat((val_total_predicted, (pred_val > 0.5).int().cpu()), 0)
                val_total_pred_prob = torch.cat((val_total_pred_prob, pred_val.cpu()), 0)

            # 计算验证集的其他性能指标（例如准确率、MCC、ROC AUC、PRC等）
            val_labels = val_total_labels.detach().cpu().numpy()
            val_predicted = val_total_predicted.detach().cpu().numpy()
            val_pred_prob = val_total_pred_prob.detach().cpu().numpy()

            val_acc = accuracy_score(val_labels, val_predicted)
            val_mcc = matthews_corrcoef(val_labels, val_predicted)
            val_roc_auc = roc_auc_score(val_labels, val_pred_prob)
            val_prc = average_precision_score(val_labels, val_pred_prob)

            # 打印验证集性能
            print('# [{}/{}] validation acc={:.5f}, mcc={:.5f}, roc_auc={:.5f}, prc={:.5f}\n'.format(
                epoch + 1, epochs, val_acc, val_mcc, val_roc_auc, val_prc), end='\r')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--batchsize', type=int, default=100, help='Number of batch_size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')  # 1e-4
    parser.add_argument('--hidden-dim', type=int, default=300, help='hidden units of FC layers (default: 256)')
    parser.add_argument('--input_dim', type=int, default=704, help='input_dim.')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')  # 0.1

    parser.add_argument('--n_filters', type=int, default=64, help='n_filters')
    parser.add_argument('--embed_dim', type=int, default=128, help='embed_dim')
    parser.add_argument('--num_features_xt', type=int, default=25, help='num_features_xt')
    parser.add_argument('--output_dim', type=int, default=128, help='output_dim')
    parser.add_argument('--max_compound_len', type=int, default=300, help='output_dim')

    parser.add_argument('--seed', type=int, default=2018, help='Random Seed')
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
    parser.add_argument('--heads', type=int, default=10, help='heads')

    parser.add_argument('--cnn_kernel_size', type=int, default=8, help='CNN Conv Layers')
    parser.add_argument('--dim', type=int, default=100, help='dim')

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