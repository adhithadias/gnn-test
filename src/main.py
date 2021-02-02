import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from utils import print_graph_specs, print_data_set, train_for_dataset, train_one_graph
from net import Net, GNNStackGraph
import torch.nn.functional as F


def get_sample_graph():
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]])
    return Data(x=x, edge_index=edge_index)


if __name__ == '__main__':
    print("Starting the application...")
    print("Is cuda available: ", torch.cuda.is_available())

    data = get_sample_graph()
    print_graph_specs(data)

    print("Moving graph data to the device")
    device = torch.device('cuda:0')
    data = data.to(device)
    print(data)

    print("\n============Enzime Graph=================\n")
    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    print_data_set(dataset)

    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    print_data_set(dataset)

    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    print_data_set(dataset)

    print("\n================== TRAIN ========================")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset.num_node_features, dataset.num_classes).to(device)

    data_size = len(dataset)
    # loader = DataLoader(dataset., batch_size=32, shuffle=True)
    # loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=64, shuffle=True)
    # test_loader = DataLoader(dataset[int(data_size * 0.8):], batch_size=64, shuffle=True)

    data = dataset[0].to(device)
    print_graph_specs(data)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()

    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    print('Accuracy: {:.4f}'.format(acc))

    # /* ----------------------------------- */

    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    train_for_dataset(dataset)

    dataset = Planetoid(root='/tmp/cora', name='cora')
    train_one_graph(dataset)




