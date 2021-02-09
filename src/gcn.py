import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from models import GCNNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data',
                dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)

model = GCNNet(dataset.num_node_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.detach().item()


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

file = 'gcn.pt'
model_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'model', file)

model.eval()
torch.save(model.state_dict(), model_path)
