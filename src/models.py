import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(input_dim, 16).jittable()
        self.conv2 = GCNConv(16, output_dim).jittable()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
