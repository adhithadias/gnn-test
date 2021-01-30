
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
