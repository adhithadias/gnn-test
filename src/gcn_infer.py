import os.path as osp
import torch

from models import GCNNet
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data',
                dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)


model = GCNNet(1433, 7).to(device)
model.eval()
file = 'gcn.pt'
model_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'model', file)
# model.load_state_dict(torch.jit.load(model_path))

model = torch.jit.load(model_path)

print("\n\n------ MODEL GRAPH--------")
print(model.graph)

print(model)

@torch.no_grad()
def test():
    model.eval()
    accuracy = []
    logits = model(data.x, data.edge_index)
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accuracy.append(acc)
    return accuracy

print(test())





