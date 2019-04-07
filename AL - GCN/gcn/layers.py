import math

import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.

    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    def forward(self, matrix1, matrix2):
        self.save_for_backward(matrix1, matrix2)
        return torch.mm(matrix1, matrix2)

    def backward(self, grad_output):
        matrix1, matrix2 = self.saved_tensors
        grad_matrix1 = grad_matrix2 = None

        if self.needs_input_grad[0]:
            grad_matrix1 = torch.mm(grad_output, matrix2.t())

        if self.needs_input_grad[1]:
            grad_matrix2 = torch.mm(matrix1.t(), grad_output)

        return grad_matrix1, grad_matrix2


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = SparseMM()(adj, support)
        if self.bias is None:
            return output
        return output + self.bias

    def __repr__(self):
        return "<%s (%s -> %s)>" % (type(self).__name__, self.in_features, self.out_features,)


class AsymmetricGCN(GraphConvolution):
    def __init__(self, in_features, out_features, bias=True):
        super(AsymmetricGCN, self).__init__(2 * in_features, out_features, bias=bias)

    def forward(self, x, adj):
        support = torch.mm(adj, x)
        support = torch.cat(torch.chunk(support, 2, dim=0), dim=1)
        # output = SparseMM(support)(self.weight)
        output = SparseMM()(support, self.weight)
        if self.bias is None:
            return output
        return output + self.bias
