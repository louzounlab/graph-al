import torch
import torch.nn as nn
import torch.nn.functional as functional
from gcn.layers import GraphConvolution, AsymmetricGCN


class GCNKipf(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCNKipf, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self._activation_func = functional.relu
        self._dropout = dropout

    def forward(self, x, adj):
        x = self._activation_func(self.gc1(x, adj))
        x = functional.dropout(x, self._dropout, training=self.training)
        x = self.gc2(x, adj)
        return functional.log_softmax(x, dim=1)


class GCN(nn.Module):
    def __init__(self, nfeat, hlayers, nclass, dropout, layer_type=None, is_regression=False):
        super(GCN, self).__init__()
        if layer_type is None:
            layer_type = GraphConvolution
        hidden_layers = [nfeat] + hlayers + [nclass]    # TODO: change last layer size if adding more layers is needed
        self._layers = nn.ModuleList([layer_type(first, second)
                                      for first, second in zip(hidden_layers[:-1], hidden_layers[1:])])
        self._activation_func = functional.relu
        self._dropout = dropout
        self._regression = is_regression
        # self._linear = nn.Linear(nclass*4, 2*nclass)
        # self._linear2 = nn.Linear(nclass*2, nclass)

    def forward(self, x, adj, get_representation=False):
        layers = list(self._layers)
        for layer in layers[:-1]:
            x = self._activation_func(layer(x, adj))
            x = functional.dropout(x, self._dropout, training=self.training)
        x = layers[-1](x, adj)
        # x = self._activation_func(self._linear(x))
        # if get_representation:
        #     return x
        # x = functional.dropout(x, 0.2, training=self.training)
        # x = self._linear2(x)
        if self._regression or get_representation:
            return x
        return functional.log_softmax(x, dim=1)


class GCNCombined(nn.Module):
    def __init__(self, nbow, nfeat, hlayers, nclass, dropout):
        super(GCNCombined, self).__init__()
        hlayers = hlayers[:]
        self.bow_layer = GraphConvolution(nbow, hlayers[0])
        hlayers[0] = (hlayers[0] * 2) + nfeat
        self.layers = nn.ModuleList([AsymmetricGCN(first, second) for first, second in zip(hlayers[:-1], hlayers[1:])])
        self.class_layer = AsymmetricGCN(hlayers[-1], nclass)

        self._activation_func = functional.relu
        self._dropout = dropout

    def forward(self, bow, feat, adj):
        x = self._activation_func(self.bow_layer(bow, adj))
        x = functional.dropout(x, self._dropout, training=self.training)
        x = torch.cat(torch.chunk(x, 2, dim=0), dim=1)
        x = torch.cat([x, feat], dim=1)

        for layer in self.layers:
            x = self._activation_func(layer(x, adj))
            x = functional.dropout(x, self._dropout, training=self.training)

        x = self.class_layer(x, adj)
        return functional.log_softmax(x, dim=1)


# hidden_layers = [500, 100, 35, nclass]
# hidden_layers = [nfeat, 70, 35, nclass]
# hidden_layers = [nfeat, nhid, nclass]

# , 500, 100, 35, nclass]
# hidden_layers = [nfeat, 70, 35, nclass]
# hidden_layers = [nfeat, nhid, nclass]
