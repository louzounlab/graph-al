import os
import pickle
import pandas as pd

import networkx as nx
import numpy as np
import torch
from scipy import sparse
from sklearn.model_selection import train_test_split
from bisect import insort

from graph_measures.features_infra.graph_features import GraphFeatures
from graph_measures.loggers import EmptyLogger, PrintLogger
from graph_measures.feature_meta import FeatureMeta
from graph_measures.features_algorithms.vertices.neighbor_nodes_histogram import nth_neighbor_calculator

DTYPE = np.float32


def add_neighbor_features(features_meta):
    new_features = features_meta.copy()
    neighbor_features = {
        "first_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(1), {"fnh", "first_neighbor"}),
        "second_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(2), {"snh", "second_neighbor"}),
    }
    new_features.update(neighbor_features)
    return new_features


def symmetric_normalization(mx):
    rowsum = np.array(mx.sum(1))
    rowsum[rowsum != 0] **= -0.5
    r_inv = rowsum.flatten()
    # r_inv = np.power(rowsum, -0.5).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    return r_mat_inv.dot(mx).dot(r_mat_inv)  # D^-0.5 * X * D^-0.5


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def handle_matrix_concat(mx, should_normalize=True):
    mx += sparse.eye(mx.shape[0])
    mx_t = mx.transpose()
    if should_normalize:
        mx = symmetric_normalization(mx)
        mx_t = symmetric_normalization(mx_t)

    return sparse.vstack([mx, mx_t])  # vstack: below, hstack: near


def handle_matrix_symmetric(mx):
    # build symmetric adjacency matrix
    mx += (mx.T - mx).multiply(mx.T > mx)
    return symmetric_normalization(mx + sparse.eye(mx.shape[0]))


def sparse_mx_to_torch_sparse_tensor(sparse_mx) -> torch.sparse.FloatTensor:
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(DTYPE)
    indices = torch.from_numpy(np.vstack((sparse_mx.row.astype(int), sparse_mx.col.astype(int)))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class LoadData:
    def __init__(self, data_name, data_path, features_meta, features_path=None, is_directed=True, is_regression=False):
        # self._data_dir = os.path.join("data_sets", data_name)
        self._data_dir = data_path
        self._features_path = features_path if features_path else os.path.join(self._data_dir, "features")
        self._name = data_name
        self._features_meta = features_meta
        df1 = pd.read_csv(os.path.join(self._data_dir, "graph_edges.txt"))
        if is_directed:
            self._gnx = nx.from_pandas_edgelist(df1, "n1", "n2", create_using=nx.DiGraph())
        else:
            self._gnx = nx.from_pandas_edgelist(df1, "n1", "n2")
        self._logger = PrintLogger("MyLogger")
        self._features_mx = None
        self._tags = {}
        self._content = None
        self._regression = is_regression
        print(str(self._name)+" was loaded")

    def build_features(self):
        gnx_ftr = GraphFeatures(self._gnx, self._features_meta, dir_path=self._features_path, logger=self._logger,
                                is_regression=self._regression)
        gnx_ftr.build(should_dump=True)  # build ALL_FEATURES
        self._features_mx = gnx_ftr.to_matrix(dtype=np.float32, mtype=np.matrix)
        # for i, node in enumerate(sorted(self._gnx)):
        #     self._gnx.node[node]['features'] = self._features_mx[i]
        print(self._features_mx.shape)

    def build_tags(self):
        self._gnx.graph["node_labels"] = []
        self._tags = []
        for node in self._gnx.node:
            typ = type(node)
            break
        with open(os.path.join(self._data_dir, "tags.txt"), "r") as f:
            # for node, line in enumerate(f):
                # tag = str(line.rstrip("\n"))
                # tag = tag[1:tag.find('/', 1)]
            for i, line in enumerate(f):
                if self._regression:
                    liner = line.split()
                    node = liner[0]
                    tags = liner[1:]
                    tag = list(map(float, tags))
                else:
                    node, tag = line.split()
                    node = typ(node)
                    try:
                        tag = int(tag)
                    except:
                        pass
                if not self._gnx.has_node(node):
                    self._gnx.add_node(node)
                self._gnx.node[node]['label'] = tag
                if not self._regression and tag not in self._gnx.graph["node_labels"]:
                    self._gnx.graph["node_labels"].append(tag)
                # if not(self._gnx.has_node(node)):
                #     self._gnx.add_node(node)
            # remove untagged nodes
            for node in sorted(self._gnx):
                if self._gnx.node[node] == {}:
                    self._gnx.remove_node(node)
                else:
                    self._tags.append(self._gnx.node[node]['label'])
            if self._regression:
                self._gnx.graph["node_labels"] = [i for i, x in enumerate(self._tags[0])]
        return None

    def build_content(self):

        self._content = []
        for node in self._gnx.node:
            typ = type(node)
            break
        with open(os.path.join(self._data_dir, "content.txt"), "r") as f:

            for i, line in enumerate(f):
                ls = line.split()
                node = typ(ls[0])
                content = ls[1:-1]
                self._gnx.node[node]['content'] = content

            for node in sorted(self._gnx):
                self._content.append(self._gnx.node[node]['content'])

        with open(os.path.join(self._data_dir, "content.pkl"), 'wb') as f:
            pickle.dump(self._content, f)

        return None

    def dump(self):
        nx.write_gpickle(self._gnx, os.path.join(self._data_dir, "gnx.pkl"))
        return None


# if "citeseer" == dataset:
#     Taking the largest
#     max_subgnx = max(nx.connected_component_subgraphs(self._gnx.to_undirected()), key=len)
#     self._gnx = self._gnx.subgraph(max_subgnx)
class GraphLoader(object):
    def __init__(self, data_set, data_dir, features_meta, is_max_connected=False, cuda_num=None, logger=None,
                 add_neighbors_features=False, is_directed=True, is_regression=False):
        super(GraphLoader, self).__init__()
        self._data_set = data_set
        self._data_path = data_dir
        self._logger = EmptyLogger() if logger is None else logger
        self._cuda_num = cuda_num
        self._features_meta = features_meta
        self._is_max_connected = is_max_connected
        self._logger.debug("Loading %s dataset...", self._data_path)
        self._regression = is_regression
        self._neighbors_f = add_neighbors_features
        if 'first_neighbor_histogram' in features_meta or 'second_neighbor_histogram' in features_meta:
            self._neighbors_f = True

        gpath = os.path.realpath(os.path.join(self._data_path, "gnx.pkl"))
        features_path = self._features_path()
        if not os.path.exists(gpath):
            ld = LoadData(self._data_set, self._data_path, self._features_meta, features_path, is_directed,
                          is_regression)
            ld.build_tags()
            # ld.build_content()
            ld.build_features()
            ld.dump()

        # self._gnx = pickle.load(open(gpath, "rb"))
        self._gnx = nx.read_gpickle(gpath)

        self._nodes_order = sorted(self._gnx)
        self._data_size = len(self._nodes_order)
        if self._regression:
            self._tags = [self._gnx.node[n]['label'] for n in self._nodes_order]
            self._labels = {i: i for i, label in enumerate(self._tags[0])}
            self._labels_list = list(self._labels.values())
            self._ident_labels = self._tags
        else:
            self._labels = {i: label for i, label in enumerate(self._gnx.graph["node_labels"])}
            self._labels_list = list(self._labels.values())
            self._tags = [self._labels_list.index(self._gnx.node[n]['label']) for n in self._nodes_order]
            self._ident_labels = self._encode_onehot_gnx()

        self._train_set = self._test_set = self._base_train_set = None
        self._train_idx = self._test_idx = self._base_train_idx = None
        self._val_idx = self._val_set = None

        if self._neighbors_f:
            self._features_meta = add_neighbor_features(self._features_meta)
            self._topo_mx = None
        else:
            self.build_features_matrix(print_time=True)

        # # load content
        # if os.path.exists(os.path.join(self._data_path, "content.pkl")):
        #     self._content = pickle.load(open(os.path.join(self._data_path, "content.pkl"), "rb"))
        # bow_mx = np.vstack([self._gnx.node[node]['content'] for node in self._nodes_order]).astype(DTYPE)
        # self._bow_mx = normalize(bow_mx)

        # Adjacency matrices
        adj = nx.adjacency_matrix(self._gnx, nodelist=self._nodes_order).astype(DTYPE)
        self._adj = handle_matrix_symmetric(adj)
        self._adj = sparse_mx_to_torch_sparse_tensor(self._adj).to_dense()
        self._adj_rt = handle_matrix_concat(adj, should_normalize=True)
        self._adj_rt = sparse_mx_to_torch_sparse_tensor(self._adj_rt).to_dense()

        print(data_set + " was successfully loaded")

    def _activate_cuda(self, *items):
        if self._cuda_num is None:
            if 1 == len(items):
                return items[0]
            return items
        if 1 == len(items):
            return items[0].cuda(self._cuda_num)
        return [x.cuda(self._cuda_num) for x in items]

    def _encode_onehot_gnx(self):  # gnx, nodes_order: list = None):
        ident = np.identity(len(self._labels))
        if self._gnx.graph.get('is_index_labels', False):
            labels_dict = {label: ident[i, :] for i, label in self._labels.items()}
        else:
            labels_dict = {i: ident[i, :] for i, label in self._labels.items()}
        return np.array(list(map(lambda n: labels_dict[self._labels_list.index(self._gnx.node[n]['label'])],
                                 self._nodes_order)), dtype=np.int32)

    def get_graph(self):
        return self._gnx.copy()

    def get_node(self, idx):
        return self._nodes_order[idx]

    def get_tag(self, node):
        return self._gnx.node[int(node)]['label']

    def get_tag_i(self, idx):
        node = self._nodes_order[idx]
        return self._gnx.node[node]['label']

    @property
    def num_labels(self):
        return len(self._labels)

    @property
    def labels_list(self):
        return list(self._labels_list)

    @property
    def labels(self):
        if self._regression:
            labels = torch.FloatTensor(self._tags)
        else:
            labels = torch.LongTensor(np.where(self._ident_labels)[1])
        # labels = self._labels
        return self._activate_cuda(labels)

    @property
    def tags(self):
        tags = torch.FloatTensor(self._tags)
        return self._activate_cuda(tags)

    @property
    def nodes_order(self):
        return self._nodes_order.copy()

    @property
    def data_size(self):
        return self._data_size

    @property
    def regression(self):
        return self._regression

    @property
    def base_train_idx(self):
        base_train_idx = torch.LongTensor(self._base_train_idx)
        return self._activate_cuda(base_train_idx)

    @property
    def train_idx(self):
        train_idx = torch.LongTensor(self._train_idx)
        return self._activate_cuda(train_idx)

    @property
    def val_idx(self):
        if self._val_idx:
            val_idx = torch.LongTensor(self._val_idx)
        else:
            return None
        return self._activate_cuda(val_idx)

    @property
    def test_idx(self):
        test_idx = torch.LongTensor(self._test_idx)
        return self._activate_cuda(test_idx)

    # @property
    # def train_set(self):
    #     train_set = torch.LongTensor(self._train_set)
    #     return self._activate_cuda(train_set)
    #
    # @property
    # def val_set(self):
    #     val_set = torch.LongTensor(self._val_set)
    #     return self._activate_cuda(val_set)
    #
    # @property
    # def test_set(self):
    #     test_set = torch.LongTensor(self._test_set)
    #     return self._activate_cuda(test_set)

    @property
    def bow_mx(self):
        bow_feat = torch.FloatTensor(self._bow_mx)
        return self._activate_cuda(bow_feat)

    @property
    def topo_mx(self):
        assert self._topo_mx is not None, "Split train required"
        topo_feat = torch.FloatTensor(self._topo_mx)
        return self._activate_cuda(topo_feat)

    @property
    def adj_rt_mx(self):
        return self._activate_cuda(self._adj_rt.clone())

    @property
    def adj_mx(self):
        return self._activate_cuda(self._adj.clone())

    # split the data to train and test
    def split_test(self, test_p, build_features=True, features_meta=None):
        if features_meta is None:
            features_meta = self._features_meta
        indexes = range(len(self._nodes_order))
        if test_p == 1:
            self._test_set = self._nodes_order.copy()
            self._test_idx = list(indexes)
            self._base_train_set = []
            self._base_train_idx = []
        else:
            self._base_train_set, self._test_set, self._base_train_idx, self._test_idx = \
                        train_test_split(self._nodes_order, indexes, test_size=test_p, shuffle=True)

        self._train_set = self._base_train_set.copy()
        self._train_idx = self._base_train_idx.copy()
        if self._neighbors_f and build_features:
            self.build_features_matrix(features_meta, self._base_train_set)

    def node_val_to_train(self, node, build_features=True):
        # node = int(node)
        insort(self._train_set, node)
        insort(self._train_idx, self._nodes_order.index(node))
        self._val_set.remove(node)
        self._val_idx.remove(self._nodes_order.index(node))

        if self._neighbors_f and build_features:
            self.build_features_matrix(self._features_meta, self._train_set)

    def idx_val_to_train(self, idx, build_features=True):
        node = self._nodes_order[idx]
        self.node_val_to_train(node, build_features=build_features)

    def node_test_to_train(self, node, build_features=True):
        insort(self._base_train_set, node)
        insort(self._base_train_idx, self._nodes_order.index(node))
        insort(self._train_set, node)
        insort(self._train_idx, self._nodes_order.index(node))
        self._test_set.remove(node)
        self._test_idx.remove(self._nodes_order.index(node))

        if self._neighbors_f and build_features:
            self.build_features_matrix(self._features_meta, self._train_set)

    def idx_test_to_train(self, idx, build_features=True):
        node = self._nodes_order[idx]
        self.node_test_to_train(node, build_features=build_features)

    def node_test_to_val(self, node, build_features=True):
        insort(self._base_train_set, node)
        insort(self._base_train_idx, self._nodes_order.index(node))
        insort(self._val_set, node)
        insort(self._val_idx, self._nodes_order.index(node))
        self._test_set.remove(node)
        self._test_idx.remove(self._nodes_order.index(node))

        if self._neighbors_f and build_features:
            self.build_features_matrix(self._features_meta, self._train_set)

    def idx_test_to_val(self, idx, build_features=True):
        node = self._nodes_order[idx]
        self.node_test_to_val(node, build_features=build_features)

    def _features_path(self):
        if self._is_max_connected:
            return os.path.join(self._data_path, "features%d" % (self._is_max_connected,))
        return os.path.join(self._data_path, "features")

    def build_features_matrix(self, features_meta=None, train_set=None, print_time=False):
        if features_meta is None:
            features_meta = self._features_meta
        if train_set is None:
            train_set = self._train_set
        if train_set is None:
            train_set = set()

        feat_mx = self.get_features(features_meta, train_set, print_time)

        # replace all nan values of attractor basin to 100
        feat_mx[np.isnan(feat_mx)] = 100

        # ratio = 10 ** np.ceil(np.log10(abs(np.mean(topo_mx) / np.mean(self._bow_mx))))
        # topo_mx /= ratio

        self._topo_mx = feat_mx
        return feat_mx

    def get_features(self, features_meta, train_set=None, print_time=False):
        if train_set is not None:
            train_set = set(train_set)
        features_path = self._features_path()
        features = GraphFeatures(self._gnx, features_meta, dir_path=features_path, is_regression=self._regression,
                                 logger=self._logger, is_max_connected=self._is_max_connected)
        features.build(include=train_set, should_dump=True, print_time=print_time)

        add_ones = bool({"first_neighbor_histogram", "second_neighbor_histogram"}.intersection(features_meta))
        feat_mx = features.to_matrix(add_ones=add_ones, dtype=np.float64, mtype=np.matrix, should_zscore=True)

        return feat_mx

    # split the train to train and validation
    def split_train(self, train_p, build_features=True, features_meta=None):
        if features_meta is None:
            features_meta = self._features_meta
        if train_p == 1:
            self._train_set = self._base_train_set.copy()
            self._train_idx = self._base_train_idx.copy()
            self._val_set = []
            self._val_idx = []
        else:
            if 1-train_p >= len(self._base_train_set):
                train_p = 0.8
            self._train_set, self._val_set, self._train_idx, self._val_idx = train_test_split(self._base_train_set,
                                                                                              self._base_train_idx,
                                                                                              test_size=1 - train_p,
                                                                                              shuffle=True)
        if self._neighbors_f and build_features:
            self.build_features_matrix(features_meta, self._train_set)

    # returns the base train and test sets
    def get_train_test(self):
        x_train = np.asmatrix([self._topo_mx[i].A1 for i in self._base_train_idx])
        y_train = [self._labels_list.index(self._gnx.node[node]['label']) for node in self._base_train_set]
        x_test = np.asmatrix([self._topo_mx[i].A1 for i in self._test_idx])
        y_test = [self._labels_list.index(self._gnx.node[node]['label']) for node in self._test_set]

        return x_train, y_train, x_test, y_test

    # returns the train and validation sets
    def get_train_val(self):
        x_train = np.asmatrix([self._topo_mx[i].A1 for i in self._train_idx])
        y_train = [self._labels_list.index(self._gnx.node[node]['label']) for node in self._train_set]
        x_val = np.asmatrix([self._topo_mx[i].A1 for i in self._val_idx])
        y_val = [self._labels_list.index(self._gnx.node[node]['label']) for node in self._val_set]

        return x_train, y_train, x_val, y_val

    # returns the train, validation, and test sets
    def get_train_val_test(self):
        x_train = np.asmatrix([self._topo_mx[i].A1 for i in self._train_idx])
        y_train = [self._labels_list.index(self._gnx.node[node]['label']) for node in self._train_set]
        x_val = np.asmatrix([self._topo_mx[i].A1 for i in self._val_idx])
        y_val = [self._labels_list.index(self._gnx.node[node]['label']) for node in self._val_set]
        x_test = np.asmatrix([self._topo_mx[i].A1 for i in self._test_idx])
        y_test = [self._labels_list.index(self._gnx.node[node]['label']) for node in self._test_set]

        return x_train, y_train, x_val, y_val, x_test, y_test

