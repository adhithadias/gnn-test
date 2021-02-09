import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from utils import print_graph_specs, print_data_set, train_for_dataset, train_one_graph
from net import GraphNet, GNNStackGraph
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision


def get_sample_graph():
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]])
    return Data(x=x, edge_index=edge_index)


def test1():
    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    print_data_set(dataset)

    print("\n================== TRAIN ========================")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphNet(dataset.num_node_features, dataset.num_classes).to(device)

    data = dataset[0].to(device)
    print_graph_specs(data)

    train_loader = test_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    print("\n\n\n\n\n")
    print(model)

    for param in model.parameters():
        print(param, param.shape)

    model.train()
    loss_list = []

    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss_list.append(loss)
        loss.backward()
        optimizer.step()

    # plt.plot(loss_list)
    # plt.xlabel('iteration')
    # plt.ylabel('loss')
    # plt.savefig("mygraph.png")

    return model, data


def train():
    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphNet(dataset.num_node_features, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    data = dataset[0].to(device)

    model.train()
    loss_list = []

    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss_list.append(loss)
        loss.backward()
        optimizer.step()

    return model, data


if __name__ == '__main__':

    # # An instance of your model.
    # model = torchvision.models.resnet18()
    #
    # # An example input you would normally provide to your model's forward() method.
    # example = torch.rand(1, 3, 224, 224)
    #
    # # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    # traced_script_module = torch.jit.trace(model, example)
    # script_module = torch.jit.script(model)
    #
    # print(traced_script_module)
    # print("\n\n\n\n")
    # print(script_module)

    ###################################################################################

    # trained_model, data_orig = train()
    # print(trained_model)
    #
    # trained_model.eval()
    # # data_infer = trained_model(data_orig.x, data_orig.edge_index).max(dim=1)
    # # print(data_infer)
    # # values, indices = data_infer
    # # print(values)
    # # print(indices)
    # #
    # # print(data_orig)
    # #
    # # # print("################################\n\n")
    # # # traced_script_module = torch.jit.trace(trained_model, data_orig)
    # # # print(traced_script_module)
    # #
    # # print("################################\n\n")
    # # script_module = torch.jit.script(trained_model)
    # # print(script_module)
    #
    # file = 'main.pt'
    # # model_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'model', file)
    #
    # torch.save(trained_model.state_dict(), "main.pt")
    #
    # trained_model.eval()
    # _, pred = trained_model(data_orig.x, data_orig.edge_index).max(dim=1)
    # correct = int(pred[data_orig.test_mask].eq(data_orig.y[data_orig.test_mask]).sum().item())
    # acc = correct / int(data_orig.test_mask.sum())
    # print('Accuracy: {:.4f}'.format(acc))

    ########################################################################################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphNet(3703, 6).to(device)
    model.load_state_dict(torch.load("main.pt"))

    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    data = dataset[0].to(device)

    model.eval()
    _, pred = model(data.x, data.edge_index).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    print('Accuracy: {:.4f}'.format(acc))

    print(model)





def test2():
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
    model = GraphNet(dataset.num_node_features, dataset.num_classes).to(device)

    data_size = len(dataset)
    # loader = DataLoader(dataset., batch_size=32, shuffle=True)
    # loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=64, shuffle=True)
    # test_loader = DataLoader(dataset[int(data_size * 0.8):], batch_size=64, shuffle=True)

    data = dataset[0].to(device)
    print_graph_specs(data)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()

    for epoch in range(200):
        print(epoch)
        optimizer.zero_grad()
        out = model(data)
        print(out)
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




