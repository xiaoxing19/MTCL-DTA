def train(model, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, lr, epoch,
          batch_size, affinity_graph, drug_pos, target_pos, drug_neg, target_neg):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    LOG_INTERVAL = 10
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, chain(model.parameters())), lr=lr, weight_decay=0)
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))
    epoch_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        output, ssl_loss = model(affinity_graph.to(device), drug_graph_batchs,target_graph_batchs, drug_pos, target_pos,data.to(device), drug_neg, target_neg)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device)) + ssl_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))



def test(model, device, loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph, drug_pos,
         target_pos, drug_neg, target_neg):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))  # drug graphs
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))  # target graphs
    with torch.no_grad():
        for data in loader:
            output, _ = model(affinity_graph.to(device), drug_graph_batchs, target_graph_batchs, drug_pos, target_pos, data, drug_neg, target_neg)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def train_predict():
    print("Data preparation in progress for the {} dataset...".format(args.dataset))
    affinity_mat = load_data(args.dataset)
    full_data, affinity_graph, drug_pos, target_pos, drug_neg, target_neg = process_data(affinity_mat, args.dataset, args.num_pos)

    drug_graphs_dict = get_drug_molecule_graph(
        json.load(open(f'data/{args.dataset}/drugs.txt'), object_pairs_hook=OrderedDict))
    drug_smiles = get_drug_smiles(
        json.load(open(f'data/{args.dataset}/drugs.txt'), object_pairs_hook=OrderedDict))

    drug_graphs_Data = GraphDataset(graphs_dict=drug_graphs_dict, dttype="drug", drug_smiles=drug_smiles)
    drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=collate,
                                                         batch_size=affinity_graph.num_drug)


    target_graphs_dict = get_target_molecule_graph(
        json.load(open(f'data/{args.dataset}/targets.txt'), object_pairs_hook=OrderedDict), args.dataset)
    target_smiles = get_target_smiles(json.load(open(f'data/{args.dataset}/targets.txt')))
    target_graphs_Data = GraphDataset(graphs_dict=target_graphs_dict, dttype="target", target_smiles=target_smiles)
    target_graphs_DataLoader = torch.utils.data.DataLoader(target_graphs_Data, shuffle=False, collate_fn=collate,
                                                           batch_size=affinity_graph.num_target)

    device = torch.device('cuda:0')

    drug_pos = drug_pos.to(device)
    target_pos = target_pos.to(device)
    drug_neg = drug_neg.to(device)
    target_neg = target_neg.to(device)
    # model.load_state_dict(torch.load('model/best_model.pt'))

    # Initialize 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_index, test_index) in enumerate(kf.split(full_data)):

        best_mse = float('inf')
        best_model_params = None  # 用于保存最好的模型参数

        model = MTCLDTA(tau=args.tau,
                        lam=args.lam,
                        ns_dims=[affinity_graph.num_drug + affinity_graph.num_target + 2, 512, 256],
                        d_ms_dims=[78, 78, 78 * 2, 256],
                        t_ms_dims=[54, 54, 54 * 2, 256],
                        embedding_dim=128,
                        dropout_rate=args.edge_dropout_rate)
        model.to(device)

        print(f"\nStarting Fold {fold + 1}")

        # Split data into train and test for this fold
        train_data = [full_data[i] for i in train_index]
        test_data = [full_data[i] for i in test_index]

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)


        print("Start training...")
        for epoch in range(args.epochs):
            train(model, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, args.lr, epoch+1,
                  args.batch_size, affinity_graph, drug_pos, target_pos, drug_neg, target_neg)
            G, P = test(model, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader,
                        affinity_graph, drug_pos, target_pos, drug_neg, target_neg)

            mse, rm2, ci = model_evaluate(G, P)
            print(f"MSE: {mse}, RM2: {rm2}, Ci: {ci}")

            # Save model if it has the best performance on the validation set
            if mse < best_mse:
                best_mse = mse
                best_model_params = model.state_dict()
                print(f"New best MSE for Fold {fold + 1}: {best_mse} at Epoch {epoch + 1}")

        # torch.save(best_model_params, f'model/best_model_fold_{fold + 1}.pt')
        torch.save(best_model_params, f'model/best_{fold + 2}.pt')
        print(f"Best model for fold {fold + 1} saved ")




if __name__ == '__main__':
    from sklearn.model_selection import KFold
    import os
    import argparse
    import torch
    import json
    import warnings
    from collections import OrderedDict
    from torch import nn
    from itertools import chain
    from data_process import load_data, process_data, get_drug_molecule_graph, get_target_molecule_graph, get_drug_smiles, get_target_smiles
    from utils import GraphDataset, collate, model_evaluate
    from model import MTCLDTA
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.data import DataLoader, random_split, ConcatDataset

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='davis')
    parser.add_argument('--epochs', type=int, default=2500)  # --kiba 3000 --davis 2500
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--edge_dropout_rate', type=float, default=0.2)  # --kiba 0.  --davis 0.2
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--num_pos', type=int, default=5)  # --kiba 10 --davis 3
    parser.add_argument('--pos_threshold', type=float, default=8.0)
    args, _ = parser.parse_known_args()
    # args = parser.parse_args()

    train_predict()




