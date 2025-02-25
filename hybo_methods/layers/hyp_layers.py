import torch
import torch.sparse as tsp
import torch.nn as nn
from torch.nn.modules.module import Module
from .conv import HypergraphConv


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, network, num_layers):
        super(HyperbolicGraphConvolution, self).__init__()
        self.agg = HypAgg(manifold, c_in, out_features, network, num_layers)

    def forward(self, input):
        x, adj = input
        h = self.agg.forward(x, adj)
        output = h, adj
        return output


class StackGCNs(Module):

    def __init__(self, num_layers):
        super(StackGCNs, self).__init__()

        self.num_gcn_layers = num_layers - 1

    def plainGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            output.append(torch.spmm(adj, output[i]))
        return output[-1]

    def resSumGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            output.append(torch.spmm(adj, output[i]))
        return sum(output[1:])

    def resAddGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        if self.num_gcn_layers == 1:
            return torch.spmm(adj, x_tangent)
        for i in range(self.num_gcn_layers):
            if i == 0:
                output.append(torch.spmm(adj, output[i]))
            else:
                output.append(output[i] + torch.spmm(adj, output[i]))
        return output[-1]

    def denseGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            if i > 0:
                output.append(sum(output[1:i + 1]) + torch.spmm(adj, output[i]))
            else:
                output.append(torch.spmm(adj, output[i]))
        return output[-1]


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, network, num_layers):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.stackGCNs = getattr(StackGCNs(num_layers), network)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        # 这里的gcn操作跟欧式空间一样,直接卷就行了
        output = self.stackGCNs((x_tangent, adj))
        output = self.manifold.proj(self.manifold.expmap0(output, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class H2GraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c, num_layers, dropout, n_heads, use_att=True):
        super(H2GraphConvolution, self).__init__()
        self.agg = H2Agg(manifold, c, in_features, out_features, num_layers, dropout, n_heads, use_att=use_att)

    def forward(self, x, adj):
        h = self.agg.forward(x, adj)
        return h


class H2Agg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, out_features, num_layers, dropout, n_heads, use_att=True):
        super(H2Agg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.num_layers = num_layers
        self.stackGCNS = nn.ModuleList()
        for i in range(num_layers):
            self.stackGCNS.append(
                HypergraphConv(
                    in_channels=in_features,
                    out_channels=out_features,
                    dropout=dropout,
                    heads=n_heads,
                    use_attention=use_att,
                    concat=False
                ))

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        # 这里的gcn操作跟欧式空间一样,直接卷就行了
        output = [x_tangent]
        for i in range(self.num_layers):
            output.append(self.stackGCNS[i](output[i], adj))
        output = sum(output) / (self.num_layers + 1)
        output = self.manifold.proj(self.manifold.expmap0(output, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


