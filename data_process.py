import pickle
import json
import torch
import os
import numpy as np
import scipy.sparse as sp
import networkx as nx

from torch_geometric import data as DATA
from collections import OrderedDict
from rdkit import Chem
from utils import DTADataset, sparse_mx_to_torch_sparse_tensor, minMaxNormalize, denseAffinityRefine
from transformers import  AutoTokenizer, AutoModelForMaskedLM, BertTokenizerFast, BertModel,BertTokenizer, AutoModel

# 将字典中的所有数值归一化，使得它们的值分布在 0 到 1 之间，同时添加一个新的键 'X'，其值为原始数值范围的中点
def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0

    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


def load_data(dataset):
    affinity = pickle.load(open('data/' + dataset + '/affinities', 'rb'), encoding='latin1')
    if dataset == 'davis':
        affinity = -np.log10(affinity / 1e9)

    return affinity

# 参数：亲和力矩阵、数据集（davis或kiba）,正样本数量、亲和力正样本阈值
def process_data(affinity_mat, dataset, num_pos):
    dataset_path = 'data/' + dataset + '/'

    train_file = json.load(open(dataset_path + 'S1_train_set.txt'))
    train_index = []
    for i in range(len(train_file)):
        train_index += train_file[i]
    test_index = json.load(open(dataset_path + 'S1_test_set.txt'))

    rows, cols = np.where(np.isnan(affinity_mat) == False)

    train_rows, train_cols = rows[train_index], cols[train_index]
    train_Y = affinity_mat[train_rows, train_cols]
    test_rows, test_cols = rows[test_index], cols[test_index]
    test_Y = affinity_mat[test_rows, test_cols]

    # 合并训练集和测试集的数据
    all_rows = np.concatenate([train_rows, test_rows])
    all_cols = np.concatenate([train_cols, test_cols])
    all_Y = np.concatenate([train_Y, test_Y])

    # 创建合并后的数据集
    all_dataset = DTADataset(drug_ids=all_rows, target_ids=all_cols, y=all_Y)

    affinity_graph, drug_pos, target_pos, drug_neg, target_neg = get_affinity_graph(dataset, affinity_mat, num_pos)

    return all_dataset, affinity_graph, drug_pos, target_pos, drug_neg, target_neg

def get_affinity_graph(dataset, adj, num_pos, alpha=0.5):
    dataset_path = 'data/' + dataset + '/'
    num_drug, num_target = adj.shape[0], adj.shape[1]

    adj = torch.tensor(adj)

    adj_matrix = torch.zeros_like(torch.tensor(adj))

    # 标记亲和力大于阀值的位置
    if dataset == "davis":
        adj_matrix = adj > 7
    elif dataset == "kiba":
        adj_matrix = adj > 12.1

    # 二元化
    adj_matrix = adj_matrix.float()

    # 药物共享靶标数
    drug_share = torch.mm(adj_matrix, adj_matrix.T)
    drug_share = drug_share.fill_diagonal_(0)
    # 获得矩阵的最大值
    drug_max = torch.max(drug_share)
    drug_min = torch.min(drug_share)
    # 最大最小规范化
    dtd = (drug_share - drug_min) / (drug_max - drug_min)
    dtd = np.nan_to_num(dtd)
    # 从文件 drug-drug-sim.txt 加载额外的药物相似性数据
    d_d = np.loadtxt(dataset_path + 'drug-drug-sim.txt', delimiter=',')
    # 与dtd矩阵加权求和
    dAll = alpha * dtd + (1 - alpha) * d_d

    drug_pos = np.zeros((num_drug, num_drug))
    drug_neg = np.zeros((num_drug, num_drug))
    # 对于 dAll 矩阵中的每个药物节点，根据其与其他药物的相似性，
    # 选择 num_pos 个最高相似性的药物作为正样本。如果相似性较高的节点数量少于 num_pos，则选择所有非零相似性的药物
    for i in range(len(dAll)):
        one = dAll[i].nonzero()[0]
        if len(one) > num_pos:
            oo = np.argsort(-dAll[i, one])
            sele = one[oo[:num_pos]]
            drug_pos[i, sele] = 1
            # 负样本
            oo_neg = np.argsort(dAll[i, one])
            sele_neg = one[oo_neg[:num_pos]]
            drug_neg[i, sele_neg] = 1
        else:
            drug_pos[i, one] = 1
            drug_neg[i, one] = 1
    drug_pos = sp.coo_matrix(drug_pos)
    drug_pos = sparse_mx_to_torch_sparse_tensor(drug_pos)
    drug_neg = sp.coo_matrix(drug_neg)
    drug_neg = sparse_mx_to_torch_sparse_tensor(drug_neg)


    # 靶标共享药物数
    target_share = torch.mm(adj_matrix.T, adj_matrix)
    target_share = target_share.fill_diagonal_(0)
    target_max = torch.max(target_share)
    target_min = torch.min(target_share)
    tdt = (target_share - target_min) / (target_max - target_min)
    tdt = np.nan_to_num(tdt)
    t_t = np.loadtxt(dataset_path + 'target-target-sim.txt', delimiter=',')
    tAll = alpha * tdt + (1 - alpha) * t_t

    target_pos = np.zeros((num_target, num_target))
    target_neg = np.zeros((num_target, num_target))
    for i in range(len(tAll)):
        one = tAll[i].nonzero()[0]
        if len(one) > num_pos:
            oo = np.argsort(-tAll[i, one])
            sele = one[oo[:num_pos]]
            target_pos[i, sele] = 1

            # 负样本
            oo_neg = np.argsort(tAll[i, one])
            sele_neg = one[oo_neg[:num_pos]]
            target_neg[i, sele_neg] = 1
        else:
            target_pos[i, one] = 1
            target_neg[i, one] = 1
    target_pos = sp.coo_matrix(target_pos)
    target_pos = sparse_mx_to_torch_sparse_tensor(target_pos)
    target_neg = sp.coo_matrix(target_neg)
    target_neg = sparse_mx_to_torch_sparse_tensor(target_neg)

    adj_1 = adj_matrix
    adj_2 = adj_matrix.T
    adj = np.concatenate((
        np.concatenate((np.zeros([num_drug, num_drug]), adj_1), 1),
        np.concatenate((adj_2, np.zeros([num_target, num_target])), 1)
    ), 0)

    train_row_ids, train_col_ids = np.where(adj != 0)
    edge_indexs = np.concatenate((
        np.expand_dims(train_row_ids, 0),
        np.expand_dims(train_col_ids, 0)
    ), 0)
    # print(edge_indexs)
    # edge_weights = adj[train_row_ids, train_col_ids]
    node_type_features = np.concatenate((
        np.tile(np.array([1, 0]), (num_drug, 1)),
        np.tile(np.array([0, 1]), (num_target, 1))
    ), 0)
    adj_features = np.zeros_like(adj)
    adj_features[adj != 0] = 1
    # print(adj_features.shape)
    features = np.concatenate((node_type_features, adj_features), 1)
    affinity_graph = DATA.Data(x=torch.Tensor(features), adj=torch.Tensor(adj),
                               edge_index=torch.LongTensor(edge_indexs))

    # affinity_graph.__setitem__("edge_weight", torch.Tensor(edge_weights))
    affinity_graph.__setitem__("num_drug", num_drug)
    affinity_graph.__setitem__("num_target", num_target)

    return affinity_graph, drug_pos, target_pos, drug_neg, target_neg


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))

    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):

    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def get_drug_molecule_graph(ligands):
    smile_graph = OrderedDict()

    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        smile_graph[d] = smile_to_graph(lg)

    return smile_graph

# 获取药物序列
def get_drug_smiles(ligands):
    drug_smiles = OrderedDict()

    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drug_emb = drug_encode_smiles(lg, r"model\drug_model")
        drug_smiles[d] = drug_emb

    return drug_smiles

def drug_encode_smiles(smiles, model_path):
    """
    将多个SMILES序列编码为嵌入
    :param smiles_list: SMILES序列列表
    :return: 嵌入Tensor
    """
    # 初始化分词器和模型
    tokenizer = BertTokenizerFast.from_pretrained(model_path)  # 选择适合的预训练模型
    model = BertModel.from_pretrained(model_path)

    # 使用分词器对SMILES序列进行编码
    inputs = tokenizer(smiles, return_tensors='pt', padding="max_length", truncation=True, max_length=100)

    with torch.no_grad():
        # 前向传播，获取模型输出
        outputs  = model(**inputs)

    # 提取最后隐藏状态的张量表示
    last_hidden_state = outputs.last_hidden_state

    return last_hidden_state



def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])

    return c_size, features, edge_index

# 获取靶标分子图
def get_target_molecule_graph(proteins, dataset):
    msa_path = 'data/' + dataset + '/aln'
    contac_path = 'data/' + dataset + '/pconsc4'    

    target_graph = OrderedDict()
    for t in proteins.keys():
        g = target_to_graph(t, proteins[t], contac_path, msa_path)
        target_graph[t] = g

    return target_graph

def get_target_smiles(proteins):

    target_smiles = OrderedDict()
    for t in proteins.keys():
        # 获取 SMILES 序列
        smiles_sequence = proteins[t]
        target_emb = target_encode_smiles(smiles_sequence, r"model\target_model")
        target_smiles[t] = target_emb

    return target_smiles

# 蛋白质bert
def target_encode_smiles(smiles, model_path):
    """
    将多个SMILES序列编码为嵌入
    :param smiles_list: SMILES序列列表
    :return: 嵌入Tensor
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # 选择适合的预训练模型
    model = AutoModel.from_pretrained(model_path)

    # 调整模型的嵌入层大小，使其与分词器的词汇表大小一致
    # model.resize_token_embeddings(len(tokenizer))

    # 使用分词器对SMILES序列进行编码
    inputs = tokenizer(smiles, return_tensors='pt', padding="max_length", truncation=True, max_length=850)

    # 删除 token_type_ids，避免传递给不需要它的模型
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']

    with torch.no_grad():
        # 前向传播，获取模型输出
        outputs  = model(**inputs)

    # 提取最后隐藏状态的张量表示
    last_hidden_state = outputs.last_hidden_state
    return last_hidden_state


# aln文件用来获取靶标序列特征，npy文件用来获取靶标分子图
def target_to_graph(target_key, target_sequence, contact_dir, aln_dir):
    target_size = len(target_sequence)
    contact_file = os.path.join(contact_dir, target_key + '.npy')

    target_feature = target_to_feature(target_key, target_sequence, aln_dir)

    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(target_size))
    index_row, index_col = np.where(contact_map >= 0.5)
    target_edge_index = []
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_edge_index = np.array(target_edge_index)

    return target_size, target_feature, target_edge_index


def target_feature(aln_file, pro_seq):
    pssm = PSSM_calculation(aln_file, pro_seq)
    # print(pssm.shape)
    other_feature = seq_feature(pro_seq)
    # print(other_feature.shape)

    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)


def target_to_feature(target_key, target_sequence, aln_dir):
    aln_file = os.path.join(aln_dir, target_key + '.aln')
    feature = target_feature(aln_file, target_sequence)

    return feature

# 这段代码的主要目的是从序列比对中提取出每个位置上各个氨基酸出现的相对频率，并将其转换为一个标准化的矩阵形式，即PSSM矩阵
def PSSM_calculation(aln_file, pro_seq):
    pfm_mat = np.zeros((len(pro_res_table), len(pro_seq)))
    with open(aln_file, 'r') as f:
        line_count = len(f.readlines())
        for line in f.readlines():
            if len(line) != len(pro_seq):
                print('error', len(line), len(pro_seq))
                continue
            count = 0
            for res in line:
                if res not in pro_res_table:
                    count += 1
                    continue
                pfm_mat[pro_res_table.index(res), count] += 1
                count += 1
    pseudocount = 0.8
    ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
    pssm_mat = ppm_mat

    return pssm_mat


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]

    return np.array(res_property1 + res_property2)


def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))

    for i in range(len(pro_seq)):
        pro_hot[i, ] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i, ] = residue_features(pro_seq[i])

    return np.concatenate((pro_hot, pro_property), axis=1)
