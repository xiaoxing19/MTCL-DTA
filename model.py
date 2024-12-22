import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, GCNConv, GATConv, global_mean_pool as gep
from torch_geometric.utils import dropout_adj
from torch.nn.utils.weight_norm import weight_norm

from transformers import BertTokenizer, BertModel
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForMaskedLM

class GCNBlock(nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(GCNBlock, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            conv_layer = GCNConv(gcn_layers_dim[i], gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, edge_index, edge_weight, batch):
        output = x
        embeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            output = self.conv_layers[conv_layer_index](output, edge_index, edge_weight)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(gep(output, batch))

        return embeddings


class GCNModel(nn.Module):
    def __init__(self, layers_dim):
        super(GCNModel, self).__init__()

        self.num_layers = len(layers_dim) - 1
        self.graph_conv = GCNBlock(layers_dim, relu_layers_index=list(range(self.num_layers)))

    def forward(self, graph_batchs):
        embedding_batchs = list(
                map(lambda graph: self.graph_conv(graph.x, graph.edge_index, None, graph.batch), graph_batchs))
        embeddings = []
        for i in range(self.num_layers):
            embeddings.append(torch.cat(list(map(lambda embedding_batch: embedding_batch[i], embedding_batchs)), 0))

        return embeddings


class DenseGCNBlock(nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(DenseGCNBlock, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            conv_layer = DenseGCNConv(gcn_layers_dim[i], gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, adj):
        output = x
        embeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            output = self.conv_layers[conv_layer_index](output, adj, add_loop=False)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(torch.squeeze(output, dim=0))

        return embeddings


class DenseGCNModel(nn.Module):
    def __init__(self, layers_dim, edge_dropout_rate=0.):
        super(DenseGCNModel, self).__init__()

        self.edge_dropout_rate = edge_dropout_rate
        self.num_layers = len(layers_dim) - 1
        self.graph_conv = DenseGCNBlock(layers_dim, 0.1, relu_layers_index=list(range(self.num_layers)),
                                        dropout_layers_index=list(range(self.num_layers)))

    def forward(self, graph):
        xs, adj, num_d, num_t = graph.x, graph.adj, graph.num_drug, graph.num_target

        indexs = torch.where(adj != 0)
        edge_indexs = torch.cat((torch.unsqueeze(indexs[0], 0), torch.unsqueeze(indexs[1], 0)), 0)
        edge_indexs_dropout, edge_weights_dropout = dropout_adj(edge_index=edge_indexs, edge_attr=adj[indexs],
                                                                p=self.edge_dropout_rate, force_undirected=True,
                                                                num_nodes=num_d + num_t, training=self.training)
        adj_dropout = torch.zeros_like(adj)
        adj_dropout[edge_indexs_dropout[0], edge_indexs_dropout[1]] = edge_weights_dropout

        embeddings = self.graph_conv(xs, adj_dropout)

        return embeddings

class DenseGCNModelSeman(nn.Module):
    def __init__(self, layers_dim, edge_dropout_rate=0.):
        super(DenseGCNModelSeman, self).__init__()

        self.edge_dropout_rate = edge_dropout_rate
        self.num_layers = len(layers_dim) - 1
        self.graph_conv = DenseGCNBlock(layers_dim, 0.1, relu_layers_index=list(range(self.num_layers)),
                                        dropout_layers_index=list(range(self.num_layers)))

    def forward(self, graph):
        xs, adj, num_d, num_t = graph.x, graph.adj, graph.num_drug, graph.num_target
        # 二阶邻接矩阵
        adj_2 = torch.matmul(adj, adj)

        indexs = torch.where(adj_2 != 0)

        edge_indexs = torch.cat((torch.unsqueeze(indexs[0], 0), torch.unsqueeze(indexs[1], 0)), 0)
        edge_indexs_dropout, edge_weights_dropout = dropout_adj(edge_index=edge_indexs, edge_attr=adj_2[indexs],
                                                                p=self.edge_dropout_rate, force_undirected=True,
                                                                num_nodes=num_d + num_t, training=self.training)
        adj_dropout = torch.zeros_like(adj_2)
        adj_dropout[edge_indexs_dropout[0], edge_indexs_dropout[1]] = edge_weights_dropout

        embeddings = self.graph_conv(xs, adj_dropout)

        return embeddings

class LinearBlock(nn.Module):
    def __init__(self, linear_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(LinearBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(linear_layers_dim) - 1):
            layer = nn.Linear(linear_layers_dim[i], linear_layers_dim[i + 1])
            self.layers.append(layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x):
        output = x
        embeddings = []
        for layer_index in range(len(self.layers)):
            output = self.layers[layer_index](output)
            if layer_index in self.relu_layers_index:
                output = self.relu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(output)

        return embeddings




class Contrast(nn.Module):
    def __init__(self, hidden_dim, output_dim, tau, lam):
        super(Contrast, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim))
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)

        return sim_matrix

    def forward(self, za, zb, pos, neg):
        za_proj = self.proj(za)
        zb_proj = self.proj(zb)
        matrix_a2b = self.sim(za_proj, zb_proj)
        matrix_b2a = matrix_a2b.t()

        # 正样本的对比损失
        matrix_a2b = matrix_a2b / (torch.sum(matrix_a2b, dim=1).view(-1, 1) + 1e-8)
        lori_a = -torch.log(matrix_a2b.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_b2a = matrix_b2a / (torch.sum(matrix_b2a, dim=1).view(-1, 1) + 1e-8)
        lori_b = -torch.log(matrix_b2a.mul(pos.to_dense()).sum(dim=-1)).mean()

        # 负样本的对比损失
        matrix_a2b_neg = matrix_a2b / (torch.sum(matrix_a2b, dim=1).view(-1, 1) + 1e-8)
        matrix_b2a_neg = matrix_b2a / (torch.sum(matrix_b2a, dim=1).view(-1, 1) + 1e-8)

        # 计算负样本损失
        lori_a_neg = -torch.log(matrix_a2b_neg.mul(neg.to_dense()).sum(dim=-1)).mean()
        lori_b_neg = -torch.log(matrix_b2a_neg.mul(neg.to_dense()).sum(dim=-1)).mean()

        # 正样本和负样本的损失
        return (self.lam * (lori_a + lori_b) + (1 - self.lam) * (lori_a_neg + lori_b_neg),
                torch.cat((za_proj, zb_proj),1))


class MTCLDTA(nn.Module):
    def __init__(self, tau, lam, ns_dims, d_ms_dims, t_ms_dims, embedding_dim=128, dropout_rate=0.2):
        super(MTCLDTA, self).__init__()

        self.output_dim = embedding_dim * 2
        self.affinity_graph_seman = DenseGCNModelSeman ([ns_dims[0], 256], dropout_rate)
        self.affinity_graph_struc = DenseGCNModel([ns_dims[0], 256], dropout_rate)

        self.drug_graph_conv = GCNModel(d_ms_dims)
        self.target_graph_conv = GCNModel(t_ms_dims)
        self.drug_contrast = Contrast(ns_dims[-1], embedding_dim, tau, lam)
        self.target_contrast = Contrast(ns_dims[-1], embedding_dim, tau, lam)


        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(320, 256)
        self.fc3 = nn.Linear(512, 256)


        self.proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )


        self.mlp = LinearBlock([1024, 512, 256, 1], 0.1, relu_layers_index=[0, 1], dropout_layers_index=[0, 1])


    def forward(self, affinity_graph, drug_graph_batchs, target_graph_batchs, drug_pos, target_pos, data, drug_neg, target_neg):
        num_d = affinity_graph.num_drug


        # BERT药物靶标序列嵌入
        drug_feature = list(map(lambda drug_graph: drug_graph.drug_feature, drug_graph_batchs))
        drug_smiles_embedding = self.fc1(torch.mean(drug_feature[0], dim=1))

        target_feature = list(map(lambda target_graph: target_graph.target_feature, target_graph_batchs))
        target_smiles_embedding = self.fc2(torch.mean(target_feature[0], dim=1))

        # affinity_graph_embedding = self.affinity_graph_conv(affinity_graph)[-1]
        affinity_graph_stru = self.affinity_graph_struc(affinity_graph)[-1]
        affinity_graph_seman = self.affinity_graph_seman(affinity_graph)[-1]
        drug_graph_embedding = self.drug_graph_conv(drug_graph_batchs)[-1]
        target_graph_embedding = self.target_graph_conv(target_graph_batchs)[-1]

        drug_embedding = torch.cat((drug_graph_embedding, drug_smiles_embedding), dim=1)
        target_embedding = torch.cat((target_graph_embedding, target_smiles_embedding), dim=1)
        drug_embedding = self.proj(drug_embedding)
        target_embedding = self.proj(target_embedding)

        dru_loss1, drug_struc_embedding = self.drug_contrast(affinity_graph_stru[:num_d], drug_embedding, drug_pos, drug_neg)
        dru_loss2, drug_seman_embedding = self.drug_contrast(affinity_graph_seman[:num_d], drug_embedding, drug_pos, drug_neg)
        tar_loss1, target_struc_embedding = self.target_contrast(affinity_graph_stru[num_d:], target_embedding, target_pos, target_neg)
        tar_loss2, target_seman_embedding = self.target_contrast(affinity_graph_seman[num_d:], target_embedding, target_pos, target_neg)

        dru_intra_loss, _ = self.drug_contrast(affinity_graph_stru[:num_d], affinity_graph_seman[:num_d], drug_pos, drug_neg)
        tar_intra_loss, _ = self.target_contrast(affinity_graph_stru[num_d:], affinity_graph_seman[num_d:], target_pos, target_neg)

        ssl_loss =  dru_loss1 + tar_loss1 + dru_loss2 + tar_loss2 + dru_intra_loss + tar_intra_loss
        drug_embedding = torch.cat((drug_struc_embedding, drug_seman_embedding), dim=1)
        target_embedding = torch.cat((target_struc_embedding, target_seman_embedding), dim=1)

        drug_id, target_id, y = data.drug_id, data.target_id, data.y
        drug_feature = drug_embedding[drug_id.int().cpu().numpy()]
        target_feature = target_embedding[target_id.int().cpu().numpy()]

        concat_feature = torch.cat((drug_feature, target_feature),dim=1)

        mlp_embeddings = self.mlp(concat_feature)

        out = mlp_embeddings[-1]

        return out, ssl_loss




