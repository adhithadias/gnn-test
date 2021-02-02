import torch
from torch_geometric.data import DataLoader
from net import Net, GNNStackGraph, GNNStackNode

def print_data_set(dataset=None):

    print("\n========== Printing dataset specs =============\n")

    if dataset is None:
        print("Null dataset..")
        return

    print("length: ", len(dataset))
    print(dataset)
    print("num classes: ", dataset.num_classes)
    print("num node features: ", dataset.num_node_features)

    for i in range(1):
        print_graph_specs(dataset[i])


def print_graph_specs(graph=None):

    print("\n========== Printing graph specs =============")

    if graph is None:
        print("Null graph..")
        return

    print(graph)
    print(graph.keys)
    print(graph['x'])

    for key, item in graph:
        print("{} found in data".format(key))

    print("Number of nodes: ", graph.num_nodes)
    print("Number of node features: ", graph.num_node_features)

    print("Number of edged: ", graph.num_edges)
    print("Number of edge features: ", graph.num_edge_features)

    print("Isolated nodes: ", graph.contains_isolated_nodes())
    print("Is directed: ", graph.is_directed())
    print("Contains self loops: ", graph.contains_self_loops())

def train_one_graph(dataset):

    test_loader = train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # build model
    model = GNNStackNode(max(dataset.num_node_features, 1), 32, dataset.num_classes, task='node')
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    # train
    for epoch in range(200):
        total_loss = 0
        model.train()
        for batch in train_loader:
            #print(batch.train_mask, '----')
            opt.zero_grad()
            embedding, pred = model(batch)
            label = batch.y
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            print("nnnnnnnnnnnnnnnnnnnnnnnnnnnnn:        ", batch.num_graphs)
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(train_loader.dataset)
        print("loss", total_loss, epoch)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(
                epoch, total_loss, test_acc))
            print("test accuracy", test_acc, epoch)

    return model


def train_for_dataset(dataset):
    dataset = dataset.shuffle()

    data_size = len(dataset)
    train_loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset[int(data_size * 0.8):], batch_size=64, shuffle=True)

    # build model
    model = GNNStackGraph(max(dataset.num_node_features, 1), 32, dataset.num_classes, task='graph')
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    # train
    for epoch in range(200):
        total_loss = 0
        model.train()
        for batch in train_loader:
            #print(batch.train_mask, '----')
            opt.zero_grad()
            embedding, pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs

        total_loss /= len(train_loader.dataset)
        print("train accuracy: ", total_loss, epoch)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(
                epoch, total_loss, test_acc))
            print("test accuracy: ", test_acc, epoch)

    return model


def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            emb, pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y

        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask
            # node classification: only evaluate on nodes in test set
            pred = pred[mask]
            label = data.y[mask]

        correct += pred.eq(label).sum().item()

    if model.task == 'graph':
        total = len(loader.dataset)
    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item()
    return correct / total
