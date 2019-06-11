import random
import timeit

from scipy.spatial.distance import cdist
import numpy as np
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from scipy.stats import entropy
import xgboost as xgb
from sklearn.model_selection import train_test_split
from neural_network.nn_activator import FeedForwardNet
from neural_network.nn_models import NeuralNet
import networkx as nx


class DistanceCalculator:
    # def __init__(self, mx, typ="euclidean"):
    #     self._mx = mx
    #     self._type = typ

    @staticmethod
    def metric(mx1, mx2, typ="euclidean", inv_cov=None, average_distance=False, scores=False):
        # Get distances as 2D array
        if typ == 'mahalanobis':
            if inv_cov is None:
                inv_cov = np.linalg.pinv(np.cov(np.vstack([mx1, mx2]).T))
            dists = cdist(mx1, mx2, typ, VI=inv_cov)
        else:
            dists = cdist(mx1, mx2, typ)
        if scores:
            if average_distance:
                means = dists.mean(axis=1)
                return np.array([np.array([dists[i][j] for j in range(len(dists[i]))
                                           if dists[i][j] < means[i]]).mean() for i in range(len(dists))])

            # the distance to the closest train node
            return dists.min(axis=1)
        # return the most distant rows
        return np.unravel_index(dists.argmax(), (mx1.shape[0], mx2.shape[0]))

    # TODO: check more outlier detection methods, https://scikit-learn.org/stable/modules/outlier_detection.html
    @staticmethod
    def one_class_learning(x_train, x_test, typ='one_class_svm', scores=False):
        if typ == 'isolation_forest':
            model = IsolationForest(behaviour='new')
        elif typ == 'local_outlier_factor':
            model = LocalOutlierFactor(novelty=True)
        elif typ == 'robust_covariance':
            model = EllipticEnvelope()
        else:
            model = svm.OneClassSVM(gamma='scale')

        model.fit(x_train)
        if scores:
            outlier_scores = 1 - model.decision_function(x_test)
            return outlier_scores
        outlier_scores = 1 - model.score_samples(x_test)
        return outlier_scores.argmax()

        # pred = model.predict(x_test)
        # indices = np.where(pred == -1)[0]
        # if len(indices) >= 1:
        #     random_idx = np.random.randint(0, len(indices))
        #     return indices[random_idx]
        # else:
        #     return np.random.randint(0, len(x_test))


# TODO: create father class - Learning
def machine_learning(x_train, y_train, x_test, batch_size=10, clf=None, is_stand_ml=False, is_al=False, is_f1=False):
    if clf is None:
        clf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
    clf.fit(np.asmatrix(x_train, dtype=np.float32), y_train)
    probs = clf.predict_proba(x_test)
    pred = clf.predict(x_test)
    if is_stand_ml:
        return pred
    if is_f1:
        # sorted_prob = [sorted(probs[i], reverse=True) for i in range(len(probs))]
        # uncertainty = [sorted_prob[i][0] - sorted_prob[i][1] for i in range(len(sorted_prob))]
        entropies = [entropy(probs[i]) for i in range(len(probs))]
        return np.argmax(entropies)
    if is_al:
        certainty = [abs(a-b) for (a, b) in probs]
        return np.argpartition(certainty, -batch_size)[-batch_size:]
    # returns the batch_size highest probability indexes
    return np.argpartition(probs[:, 1], -batch_size)[-batch_size:]


def xgb_learning(x_data, y_data, x_test, batch_size=10, n_classes=2, is_stand_ml=False, is_al=False, is_f1=False):
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, train_size=0.85, random_state=0)

    dtrain = xgb.DMatrix(x_train, label=y_train, silent=True)
    dtest = xgb.DMatrix(x_test, silent=True)
    deval = xgb.DMatrix(x_val, label=y_val, silent=True)
    params = {'silent': True, 'booster': 'dart', 'tree_method': 'auto',
              'max_depth': 7, 'lambda': 1.3, 'eta': 0.3, 'rate_drop': 0.2, 'gamma': 3,
              'sample_type': 'weighted', 'normalize_type': 'forest',
              'objective': 'multi:softprob', 'num_class': n_classes}

    clf_xgb = xgb.train(params, dtrain=dtrain, evals=[(dtrain, 'train'), (deval, 'eval')],
                        early_stopping_rounds=10, verbose_eval=False)
    probs = clf_xgb.predict(dtest)
    pred = [np.argmax(i) for i in probs]

    # clf = xgb.XGBClassifier()
    # clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=0, early_stopping_rounds=10)   # , eval_metric="auc")
    # clf.fit(x_data, y_data, verbose=0)
    # probs = clf.predict_proba(x_test)
    # pred = clf.predict(x_test)
    # correct_xg = [1 if pred[i] == y_data[i] else 0 for i in range(len(pred))]
    # acc = sum(correct_xg) / len(correct_xg)

    # param = {'booster': 'gbtree', 'max_depth': 0, 'eta': 1, 'silent': 1, 'objective': 'multi:softprob'}

    if is_stand_ml:
        return pred
    if is_f1:
        entropies = [entropy(probs[i]) for i in range(len(probs))]
        return np.argmax(entropies)
    if is_al:
        certainty = [abs(a-b) for (a, b) in probs]
        return np.argpartition(certainty, -batch_size)[-batch_size:]
    # returns the batch_size highest probability indexes
    return np.argpartition(probs[:, 1], -batch_size)[-batch_size:]


# binary deep learning
class DeepLearning:
    def __init__(self, x_shape):
        from keras import Sequential
        from keras.callbacks import EarlyStopping
        from keras.layers import Dense, Dropout
        from keras.regularizers import l1_l2
        self.early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, mode='min', verbose=1, )
        self.classifier = Sequential()
        self.classifier.add(Dense(300, kernel_initializer="he_normal", activation="elu", input_dim=x_shape))
        self.classifier.add(Dropout(0.3))
        self.classifier.add(Dense(450, kernel_initializer='he_normal', activation='elu'))
        self.classifier.add(Dropout(0.3))
        self.classifier.add(Dense(100, kernel_initializer='he_normal', activation='elu'))
        self.classifier.add(Dropout(0.3))
        self.classifier.add(Dense(20, kernel_initializer='he_normal', activation='elu', kernel_regularizer=l1_l2()))
        self.classifier.add(Dense(1, kernel_initializer='uniform', activation="sigmoid",
                                  activity_regularizer=l1_l2(0.005, 0.005)))

        self.classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # TODO: implementation using keras
    def learn(self, x_train, y_train, x_test, batch_size=10, is_stand_ml=False, is_al=False):
        self.classifier.fit(x_train, y_train, validation_split=0.1, callbacks=[self.early_stopping], epochs=100,
                            batch_size=64, verbose=0)
        probs = self.classifier.predict_proba(x_test)

        if is_stand_ml:
            return probs
        if is_al:
            certainty = [abs(a-0.5) for a in probs]
            return np.argpartition(certainty, -batch_size)[-batch_size:]
        return np.argpartition(probs[:, 1], -batch_size)[-batch_size:]


def aggregate(res_list):
    aggregated = {}
    for cur_res in res_list:
        for train_s, res in cur_res.items():
            for name, vals in res.items():
                if name not in aggregated:
                    aggregated[name] = {}
                if train_s not in aggregated[name]:
                    aggregated[name][train_s] = {}
                for par, score in vals.items():
                    if par not in aggregated[name][train_s]:
                        aggregated[name][train_s][par] = []
                    aggregated[name][train_s][par].append(score)
    return aggregated


class StandardML:
    def __init__(self, features_mx, tags_vect, label=0):
        self._tags = tags_vect
        self._features_mx = features_mx
        counter_class = Counter(tags_vect)
        self._n_class = len(counter_class)
        self._smallest_class = counter_class.most_common()[-1-label][0]
        # self._largest_class = counter_class.most_common()[0][0]
        self._n_black = counter_class.most_common()[-1-label][1]
        self._tags1 = [1 if y == self._smallest_class else 0 for y in self._tags]
        self._time = 0
        self._num_nodes = len(tags_vect)

    def _init(self, train_size=0.2, recall=True):
        # initialize the train objects with copies
        self.x_data = type(self._features_mx)(self._features_mx)
        if recall:
            self.y_data = list(self._tags1)
        else:
            self.y_data = list(self._tags)
        # split to train and test
        self.y_train = []
        while len(Counter(self.y_train)) <= 1:
            indices = list(range(self._num_nodes))
            split = int(np.round(self._num_nodes * train_size))
            train_idx = np.random.choice(indices, size=split, replace=False)
            test_idx = list(set(indices) - set(train_idx))
            self.x_train = np.vstack([self.x_data[i] for i in train_idx])
            self.y_train = [self.y_data[i] for i in train_idx]
            self.x_test = np.vstack([self.x_data[i] for i in test_idx])
            self.y_test = [self.y_data[i] for i in test_idx]

    def recall_per_data(self, train_size=0.2, certainty_rate=0.5):
        if train_size == 1:
            return 1, 1, 1, 1
        if train_size == 0:
            return 0, 0, 0, 0

        self._init(train_size=train_size)
        probs_ml, classes = machine_learning(self.x_train, self.y_train, self.x_test, is_stand_ml=True)
        # revealing the true label of all nodes which classified as black
        tags = [self.y_test[i] for i in np.where(probs_ml[:, 1] >= certainty_rate)[0]]

        black_found = sum(self.y_train) + sum(tags)
        nodes_revealed = len(self.y_train) + len(tags)
        ml_recall = black_found / self._n_black
        ml_steps = nodes_revealed / self._num_nodes

        rand_recall = sum(self.y_train) / self._n_black
        rand_steps = len(self.y_train) / self._num_nodes

        return ml_recall, ml_steps, rand_recall, rand_steps

    def run_acc(self, train_size=0.8, print_train=False, only_acc=True):
        self._init(train_size=train_size, recall=False)

        return check_scores(self.x_train, self.y_train, self.x_test, self.y_test, n_classes=self._n_class,
                            print_train=print_train, only_acc=only_acc)


def check_scores(x_train, y_train, x_test, y_test, n_classes=2, print_train=False, only_acc=False, class_weights=None):
    # XGBoost
    pred_xgb = xgb_learning(x_train, y_train, x_test, n_classes=n_classes, is_stand_ml=True)
    acc_xgb = accuracy_score(pred_xgb, y_test)
    # acc_xgb = 0

    # Random Forest
    pred_rf = machine_learning(x_train, y_train, x_test, is_stand_ml=True)
    acc_rf = accuracy_score(pred_rf, y_test)
    # acc_rf = 0

    # Neural Network
    l1 = x_train.shape[1]
    layers = (l1, int(l1*2), int(l1/2))
    net = FeedForwardNet(NeuralNet(classes=n_classes, layers_dim=layers, lr=0.01, activation_func='relu',
                                   drop_out=0.3, l2_penalty=0.001), class_weights=class_weights, gpu=True)
    # net = FeedForwardNet(NeuralNet(classes=n_classes, layers_dim=layers), gpu=False)
    net.set_data(x_train, y_train)
    net.train(total_epoch=500, stop_loss=True)
    pred_dl = net.predict(x_test)
    acc_dl = accuracy_score(pred_dl, y_test)

    print("accuracy is:  XGB - {}   RF - {}   NN - {}".format(acc_xgb, acc_rf, acc_dl))

    if print_train:
        pred_xgb = xgb_learning(x_train, y_train, x_train, n_classes=n_classes, is_stand_ml=True)
        train_acc_xgb = accuracy_score(pred_xgb, y_train)

        pred_rf = machine_learning(x_train, y_train, x_train, is_stand_ml=True)
        train_acc_rf = accuracy_score(pred_rf, y_train)

        pred_dl = net.predict(x_train)
        train_acc_dl = accuracy_score(pred_dl, y_train)

        print("train accuracy is:  XGB - " + str(train_acc_xgb) +
              "   RF - " + str(train_acc_rf) + "   NN - " + str(train_acc_dl))

    if not only_acc:
        scores = {'XGB': {'acc': acc_xgb}, 'RF': {'acc': acc_rf}, 'NN': {'acc': acc_dl}}
        scores['XGB']['mic_f1'] = f1_score(y_test, pred_xgb, average='micro')
        scores['XGB']['mac_f1'] = f1_score(y_test, pred_xgb, average='macro')
        scores['RF']['mic_f1'] = f1_score(y_test, pred_rf, average='micro')
        scores['RF']['mac_f1'] = f1_score(y_test, pred_rf, average='macro')
        # scores['XGB']['mic_f1'] = 0
        # scores['XGB']['mac_f1'] = 0
        # scores['RF']['mic_f1'] = 0
        # scores['RF']['mac_f1'] = 0
        scores['NN']['mic_f1'] = f1_score(y_test, pred_dl, average='micro')
        scores['NN']['mac_f1'] = f1_score(y_test, pred_dl, average='macro')

        # print(scores)
        # print(Counter(y_test), Counter(pred_xgb), Counter(pred_rf), Counter(pred_dl))
        # print(np.where(y_train==1), np.where(np.array(pred_xgb)==1), np.where(pred_rf==1), np.where(np.array(pred_dl)==1))
        return scores

    return acc_xgb, acc_rf, acc_dl


class GraphNeighbors:
    # didn't implement directed neighbors yet
    def __init__(self, gnx: nx.Graph, direct=False):
        self._gnx = gnx
        self._direct = gnx.is_directed() and direct

    def _get_node_neighbors_with_types(self, node):
        if self._direct:
            for in_edge in self._gnx.in_edges(node):
                yield ("i", in_edge[0])

            for out_edge in self._gnx.out_edges(node):
                yield ("o", out_edge[1])
        else:
            for edge in self._gnx.edges(node):
                yield (edge[1])

    def _iter_nodes_of_order(self, node, order: int):
        if self._direct:
            if 0 >= order:
                yield [], node
                return
            for r_type, neighbor in self._get_node_neighbors_with_types(node):
                for r_type2, neighbor2 in self._iter_nodes_of_order(neighbor, order - 1):
                    yield ([r_type] + r_type2, neighbor2)
        else:
            if 0 >= order:
                yield node
                return
            for neighbor in self._get_node_neighbors_with_types(node):
                for neighbor2 in self._iter_nodes_of_order(neighbor, order - 1):
                    yield (neighbor2)

    def get_neighbors(self, node, second_order=True, self_node=True, only_out=False):
        history = {}

        for edge in self._gnx.edges(node):
            neighbor = edge[1]
            history[neighbor] = 1
            if second_order:
                for edge2 in self._gnx.edges(neighbor):
                    neighbor2 = edge2[1]
                    if neighbor2 not in history:
                        history[neighbor2] = 2
                for edge2 in self._gnx.in_edges(neighbor):
                    neighbor2 = edge2[0]
                    if neighbor2 not in history:
                        history[neighbor2] = 2
        if not only_out:
            for edge in self._gnx.in_edges(node):
                neighbor = edge[0]
                history[neighbor] = 1
                if second_order:
                    for edge2 in self._gnx.edges(neighbor):
                        neighbor2 = edge2[1]
                        if neighbor2 not in history:
                            history[neighbor2] = 2
                    for edge2 in self._gnx.in_edges(neighbor):
                        neighbor2 = edge2[0]
                        if neighbor2 not in history:
                            history[neighbor2] = 2

        history[node] = 0
        if not self_node:
            del history[node]

        neighbors = list(history.keys())
        orders = list(history.values())
        return neighbors, orders

    def neighbors(self, second_order=True, self_node=True, with_orders=True, only_out=False):
        neighbors_matrix = []
        nodes_order = sorted(self._gnx)
        for node in nodes_order:
            neighbors, orders = self.get_neighbors(node, second_order=second_order, self_node=self_node,
                                                   only_out=only_out)
            indices = [nodes_order.index(node) for node in neighbors]
            if with_orders:
                neighbors_matrix.append(np.asmatrix([indices, orders]))
            else:
                neighbors_matrix.append(indices)

        return neighbors_matrix


def k_truss(gnx: nx.Graph):
    # g2 = nx.convert_node_labels_to_integers(gnx, ordering='sorted')
    gnx.remove_edges_from(gnx.selfloop_edges())
    nodes_order = sorted(gnx)
    nodes_map = {nodes_order[i]: i for i in range(len(nodes_order))}
    k_truss = np.zeros(len(nodes_order))
    max_k_truss = 0
    k = 1
    k_core = nx.k_core(gnx, k)
    while not nx.is_empty(k_core) and max_k_truss == k-1:
        flag = True
        while flag:
            flag = False
            # a = nx.adj_matrix(k_core).tocsr()
            # a_square = a*a
            edges_to_remove = []
            for node in sorted(k_core):
                for edge in k_core.out_edges(node):
                    node2 = edge[1]
                    # if a_square[node2, node] < k-2:
                    if len(list(nx.algorithms.simple_paths.all_simple_paths(gnx, node2, node, 2))) < k - 2:
                        edges_to_remove.append(edge)
            if len(edges_to_remove) > 0:
                k_core.remove_edges_from(edges_to_remove)
                flag = True
        for edge in k_core.out_edges:
            max_k_truss = k
            k_truss[nodes_map[edge[0]]] = k

        k += 1
        k_core = nx.k_core(gnx, k)

    return k_truss


def average_dist_from_random_set(gnx: nx.Graph, stop=0.8, iterations=1, change_to_undirect=True, batch_size=1):
    if change_to_undirect:
        gnx = gnx.to_undirected()
    gnx = nx.convert_node_labels_to_integers(gnx)

    length = dict(nx.all_pairs_shortest_path_length(gnx))
    max_length = max([max(val.values()) for val in length.values()])
    g_size = len(gnx.nodes())
    a = np.ones([g_size, g_size]) * (max_length + 1)

    for n1 in length.keys():
        for n2 in length[n1].keys():
            a[n1][n2] = length[n1][n2]

    results = []

    for i in range(iterations):
        start_time = timeit.default_timer()
        unlabeled = set((gnx.nodes()))
        labeled = []
        res = []
        for _ in range(int(stop * g_size / batch_size)):
            node = random.sample(unlabeled, batch_size)
            labeled.extend(node)
            unlabeled.difference_update(node)

            min_len = [min(a[n1, labeled]) for n1 in unlabeled]
            res.append(np.average(min_len))
        results.append(res)
        stop_time = timeit.default_timer()
        print('finish {} iteration, time:{}'.format(i, stop_time - start_time))

    results = np.array(results).mean(axis=0)
    percents = [(i + 1) * batch_size / g_size for i in range(len(results))]

    return percents, results


