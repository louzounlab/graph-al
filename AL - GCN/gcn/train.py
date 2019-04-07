from __future__ import division
from __future__ import print_function

import argparse
import logging
import random
import time

import numpy as np
import torch
import torch.nn.functional as functional
import torch.optim as optim
from torch.autograd import Variable

from graph_measures.feature_meta import NODE_FEATURES
from graph_measures.features_algorithms.vertices.neighbor_nodes_histogram import nth_neighbor_calculator
from graph_measures.features_infra.feature_calculators import FeatureMeta
from gcn import *
from gcn.data__loader import GraphLoader
from gcn.layers import AsymmetricGCN
from gcn.models import GCNCombined, GCN
from graph_measures.loggers import PrintLogger, multi_logger, EmptyLogger, CSVLogger, FileLogger
from sklearn.metrics import f1_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=1,
                        help='Specify cuda device number')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')      # fast mode will cancel early stopping
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default="cora",
                        help='The dataset to use.')
    parser.add_argument('--prefix', type=str, default="",
                        help='The prefix of the products dir name.')

    args = parser.parse_args()
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not torch.cuda.is_available():
        args.cuda = None
    return args


NEIGHBOR_FEATURES = {
    "first_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(1), {"fnh", "first_neighbor"}),
    "second_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(2), {"snh", "second_neighbor"}),
}


# parameters for early stopping
After = 50      # after how many epochs start try early stopping
Strikes = 10    # how many times in a row should the validation loss increase for stopping


def accuracy(output, labels):
    if len(labels.size()) > 1:
        return torch.tensor(0)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1_scores(output, labels, micro=True, macro=False):
    if len(labels.size()) > 1:
        return torch.tensor(0)
    preds = output.max(1)[1].type_as(labels)

    if micro and not macro:
        return f1_score(labels.cpu(), preds.cpu(), average='micro')
    if macro and not micro:
        return f1_score(labels.cpu(), preds.cpu(), average='macro')

    mic_score = f1_score(labels.cpu(), preds.cpu(), average='micro')
    mac_score = f1_score(labels.cpu(), preds.cpu(), average='macro')

    return mic_score, mac_score


def get_features(config):
    if config["feat_type"] == "neighbors":
        feature_meta = NEIGHBOR_FEATURES
    elif config["feat_type"] == "features":
        feature_meta = NODE_FEATURES
    else:
        feature_meta = NODE_FEATURES.copy()
        feature_meta.update(NEIGHBOR_FEATURES)
    return feature_meta


class ModelRunner:
    def __init__(self, data_name, dataset_path, conf, logger, data_logger=None, is_regression=False):
        self._logger = logger
        self._data_logger = EmptyLogger() if data_logger is None else data_logger
        if is_regression:
            self._criterion = torch.nn.MSELoss()
        else:
            self._criterion = torch.nn.NLLLoss()
        self._conf = conf

        features_meta = get_features(conf)
        self.loader = GraphLoader(data_name, dataset_path, features_meta, is_max_connected=False,
                                  cuda_num=conf["cuda"], logger=self._logger, is_regression=is_regression)

    @property
    def logger(self):
        return self._logger

    @property
    def data_logger(self):
        return self._data_logger

    def _get_models(self):
        # bow_feat = self.loader.bow_mx
        topo_feat = self.loader.topo_mx

        # model1 = GCN(nfeat=bow_feat.shape[1],
        #              hlayers=[self._conf["kipf"]["hidden"]],
        #              nclass=self.loader.num_labels,
        #              dropout=self._conf["kipf"]["dropout"])
        # opt1 = optim.Adam(model1.parameters(), lr=self._conf["kipf"]["lr"],
        #                   weight_decay=self._conf["kipf"]["weight_decay"])
        #
        # model2 = GCNCombined(nbow=bow_feat.shape[1],
        #                      nfeat=topo_feat.shape[1],
        #                      hlayers=self._conf["hidden_layers"],
        #                      nclass=self.loader.num_labels,
        #                      dropout=self._conf["dropout"])
        # opt2 = optim.Adam(model2.parameters(), lr=self._conf["lr"], weight_decay=self._conf["weight_decay"])

        # model3 = GCN(nfeat=topo_feat.shape[1],
        #              hlayers=self._conf["multi_hidden_layers"],
        #              nclass=self.loader.num_labels,
        #              dropout=self._conf["dropout"],
        #              layer_type=None,
        #              is_regression=self.loader.regression)
        # opt3 = optim.Adam(model3.parameters(), lr=self._conf["lr"], weight_decay=self._conf["weight_decay"])

        model4 = GCN(nfeat=topo_feat.shape[1],
                     hlayers=self._conf["multi_hidden_layers"],
                     nclass=self.loader.num_labels,
                     dropout=self._conf["dropout"],
                     layer_type=AsymmetricGCN,
                     is_regression=self.loader.regression)
        opt4 = optim.Adam(model4.parameters(), lr=self._conf["lr"], weight_decay=self._conf["weight_decay"])

        return {
            # "kipf": {
            #     "model": model1, "optimizer": opt1,
            #     "arguments": [self.loader.bow_mx, self.loader.adj_mx],
            #     "labels": self.loader.labels,
            # },
            # "our_combined": {
            #     "model": model2, "optimizer": opt2,
            #     "arguments": [self.loader.bow_mx, self.loader.topo_mx, self.loader.adj_rt_mx],
            #     "labels": self.loader.labels,
            # },
            # "our_topo_sym": {
            #     "model": model3, "optimizer": opt3,
            #     "arguments": [self.loader.topo_mx, self.loader.adj_mx],
            #     "labels": self.loader.labels,
            # },
            "our_topo_asymm": {
                "model": model4, "optimizer": opt4,
                "arguments": [self.loader.topo_mx, self.loader.adj_rt_mx],
                "labels": self.loader.labels,
            },
        }

    # verbose = 0 - silent
    # verbose = 1 - print test results
    # verbose = 2 - print train for each epoch and test results
    def run(self, train_p, verbose=2, early_stopping=False, val_p=0.1):
        self.loader.split_train(1-val_p)

        models = self._get_models()

        if self._conf["cuda"] is not None:
            [model["model"].cuda(self._conf["cuda"]) for model in models.values()]

        for model in models.values():
            model["arguments"] = list(map(Variable, model["arguments"]))
            model["labels"] = Variable(model["labels"])

        if len(self.loader.train_idx) < 1:
            result = self.test(models=models, verbose=verbose)
        else:
            # Train model
            self.train(self._conf["epochs"], models=models, verbose=verbose, early_stopping=early_stopping)

            # Testing
            result = self.test(models=models, verbose=verbose)

        if verbose != 0:
            for name, val in sorted(result.items(), key=lambda x: x[0]):
                self._data_logger.info(name, train_p, val["loss"], val["acc"], val["mic_f1"], val["mac_f1"])
        return result

    def train(self, epochs, models=None, early_stopping=False, verbose=2, split_val=False):
        if not models:
            models = self._get_models()
            if self._conf["cuda"] is not None:
                [model["model"].cuda(self._conf["cuda"]) for model in models.values()]

            for model in models.values():
                model["arguments"] = list(map(Variable, model["arguments"]))
                model["labels"] = Variable(model["labels"])
        if split_val:
            self.loader.split_train(0.9)
        train_idx, val_idx = self.loader.train_idx, self.loader.val_idx
        if early_stopping and val_idx is not None and not self._conf["fastmode"]:
            for name, model_args in models.items():
                self._train_early_stop(epochs, name, model_args, train_idx, val_idx, verbose)
        else:
            for epoch in range(epochs):
                for name, model_args in models.items():
                    self._train(epoch, name, model_args, train_idx, val_idx, verbose)

        return models

    def _train(self, epoch, model_name, model_args, idx_train, idx_val, verbose=2):
        model, optimizer = model_args["model"], model_args["optimizer"]
        arguments, labels = model_args["arguments"], model_args["labels"]

        model.train()
        optimizer.zero_grad()
        output = model(*arguments)
        loss_train = self._criterion(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        # acc_train = torch.tensor(0)

        loss_train.backward()
        optimizer.step()

        if not self._conf["fastmode"] and verbose == 2:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            if idx_val is not None:
                model.eval()
                output = model(*arguments)

                loss_val = self._criterion(output[idx_val], labels[idx_val])
                acc_val = accuracy(output[idx_val], labels[idx_val])
                # acc_val = torch.tensor(0)
            else:
                loss_val = torch.tensor(0)
                acc_val = torch.tensor(0)
            self._logger.debug(model_name + ": " +
                               'Epoch: {:04d} '.format(epoch + 1) +
                               'loss_train: {:.4f} '.format(loss_train.data.item()) +
                               'acc_train: {:.4f} '.format(acc_train.data.item()) +
                               'loss_val: {:.4f} '.format(loss_val.data.item()) +
                               'acc_val: {:.4f} '.format(acc_val.data.item()))

        # return {"loss": loss_train.data.item(), "acc": acc_train.data.item()}
        return None

    def _train_early_stop(self, epochs, model_name, model_args, idx_train, idx_val, verbose=2):
        model, optimizer = model_args["model"], model_args["optimizer"]
        arguments, labels = model_args["arguments"], model_args["labels"]
        min_loss = 999
        strike = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(*arguments)
            acc_train = accuracy(output[idx_train], labels[idx_train])
            # acc_train = torch.tensor(0)
            loss_train = self._criterion(output[idx_train], labels[idx_train])

            loss_train.backward()
            optimizer.step()

            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(*arguments)

            loss_val = self._criterion(output[idx_val], labels[idx_val])
            if epoch > After - Strikes:
                min_loss = min(loss_val, min_loss)

            if verbose == 2:
                acc_val = accuracy(output[idx_val], labels[idx_val])
                # acc_val = torch.tensor(0)
                self._logger.debug(model_name + ": " +
                                   'Epoch: {:04d} '.format(epoch + 1) +
                                   'loss_train: {:.4f} '.format(loss_train.data.item()) +
                                   'acc_train: {:.4f} '.format(acc_train.data.item()) +
                                   'loss_val: {:.4f} '.format(loss_val.data.item()) +
                                   'acc_val: {:.4f} '.format(acc_val.data.item()))

            if min_loss < loss_val * 0.95:
                strike += 1
            else:
                strike = 0
            if strike >= Strikes and epoch > After:
                self._logger.debug(model_name + ": " + "early stopping after {} epochs".format(epoch+1))
                break

        # return {"loss": loss_train.data.item(), "acc": acc_train.data.item()}
        return None

    def test(self, models=None, verbose=2):
        if not models:
            models = self._get_models()
            if self._conf["cuda"] is not None:
                [model["model"].cuda(self._conf["cuda"]) for model in models.values()]

            for model in models.values():
                model["arguments"] = list(map(Variable, model["arguments"]))
                model["labels"] = Variable(model["labels"])

        test_idx = self.loader.test_idx
        if len(test_idx) == 0:
            test_idx = self.loader.val_idx
        result = {}
        for name, model_args in models.items():
            model, arguments, labels = model_args["model"], model_args["arguments"], model_args["labels"]
            model.eval()
            output = model(*arguments)
            # loss_test = functional.nll_loss(output[test_idx], labels[test_idx])
            loss_test = self._criterion(output[test_idx], labels[test_idx])
            acc_test = accuracy(output[test_idx], labels[test_idx])
            # acc_test = torch.tensor(0)
            # mic_f1_test = torch.tensor(0)
            mic_f1_test, mac_f1_test = f1_scores(output[test_idx], labels[test_idx], micro=True, macro=True)
            if verbose != 0:
                self._logger.info(name + " Test: " +
                                  "loss= {:.4f} ".format(loss_test.data.item()) +
                                  "accuracy= {:.4f}".format(acc_test.data.item()))
            result[name] = {"loss": loss_test.data.item(), "acc": acc_test.data.item(),
                            "mic_f1": mic_f1_test, "mac_f1": mac_f1_test}

        return result

    def predict_proba(self, models, test_only=True):
        test_idx = self.loader.test_idx
        result = []
        for name, model_args in models.items():
            model, arguments, labels = model_args["model"], model_args["arguments"], model_args["labels"]
            model.eval()
            output = functional.softmax(model(*arguments), 1).detach()
            if test_only:
                result = output[test_idx]
            else:
                result = output

        return result


def last_layer_representation(models):
    result = {}
    for name, model_args in models.items():
        model, arguments = model_args["model"], model_args["arguments"]
        model.eval()
        output = model(*arguments, get_representation=True)
        result[name] = output

    return result


def init_seed(seed, cuda=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda is not None:
        torch.cuda.manual_seed(seed)


def aggregate_results(res_list):
    aggregated = {}
    for cur_res in res_list:
        for name, vals in cur_res.items():
            if name not in aggregated:
                aggregated[name] = {}
            for key, val in vals.items():
                if key not in aggregated[name]:
                    aggregated[name][key] = []
                aggregated[name][key].append(val)
    return aggregated


def mean_results(res_list):
    aggregated = aggregate_results(res_list)
    result = {}
    for name, vals in aggregated.items():
        val_list = sorted(vals.items(), key=lambda x: x[0], reverse=True)
        result[name] = {key: np.mean(val) for key, val in val_list}

    return result


def execute_runner(runner, train_p, num_iter=5, early_stopping=False, val_p=0.1):
    train_p /= 100
    test_p = 1-train_p
    # val_p = test_p = (1 - train_p) / 2.
    # train_p /= (val_p + train_p)

    # # set test to 1,000 samples and train to train_p * number of classes
    # train_p = (runner.loader.num_labels * train_p) / (runner.loader.data_size - 1000)
    # test_p = 1000 / runner.loader.data_size

    runner.loader.split_test(test_p, build_features=True)
    res = [runner.run(train_p, verbose=1, early_stopping=early_stopping, val_p=val_p) for _ in range(num_iter)]
    aggregated = aggregate_results(res)
    result = {}
    for name, vals in aggregated.items():
        val_list = sorted(vals.items(), key=lambda x: x[0], reverse=True)
        result[name] = {key: np.mean(val) for key, val in val_list}
        runner.logger.info("*"*15 + "%s mean: %s", name, ", ".join("%s=%3.4f" %
                                                                   (key, np.mean(val)) for key, val in val_list))
        runner.logger.info("*"*15 + "%s std: %s", name, ", ".join("%s=%3.4f" %
                                                                  (key, np.std(val)) for key, val in val_list))

    return result


def build_model(dataset, path=None, dropout=0.6, lr=0.01, weight_decay=0.005, hidden_layers=[16], is_regression=False):

    args = parse_args()

    seed = random.randint(1, 1000000000)
    conf = {
        # "kipf": {"hidden": args.hidden, "dropout": args.dropout, "lr": args.lr, "weight_decay": args.weight_decay},
        "kipf": {"hidden": hidden_layers, "dropout": dropout, "lr": lr, "weight_decay": weight_decay},
        "hidden_layers": hidden_layers, "multi_hidden_layers": hidden_layers, "dropout": dropout, "lr": lr,
        "weight_decay": weight_decay, "feat_type": "neighbors",
        # "weight_decay": weight_decay, "feat_type": "features",
        "dataset": dataset, "epochs": args.epochs, "cuda": args.cuda, "fastmode": args.fastmode, "seed": seed}

    init_seed(conf['seed'], conf['cuda'])
    if path:
        dataset_path = path
    else:
        # dataset_path = os.path.join(PROJ_DIR, "data_sets", dataset)
        dataset_path = os.path.join(CUR_DIR, "data_sets", dataset)

    products_path = os.path.join(CUR_DIR, "logs", args.prefix + dataset, time.strftime("%Y_%m_%d_%H_%M_%S"))
    # products_path = os.path.join(dataset_path, "logs", args.prefix + dataset, time.strftime("%Y_%m_%d_%H_%M_%S"))
    if not os.path.exists(products_path):
        os.makedirs(products_path)

    logger = multi_logger([
        PrintLogger("RoysLogger", level=logging.DEBUG),
        FileLogger("results_%s" % conf["dataset"], path=products_path, level=logging.INFO),
        FileLogger("results_%s_all" % conf["dataset"], path=products_path, level=logging.DEBUG),
    ], name=None)

    data_logger = CSVLogger("results_%s" % conf["dataset"], path=products_path)
    data_logger.info("model_name", "train_p", "loss", "acc", "mic_f1", "mac_f1")

    runner = ModelRunner(dataset, dataset_path, conf, logger=logger, data_logger=data_logger, is_regression=is_regression)

    return runner


def run_model(dataset, path=None, is_regression=False, train_size={70},
              dropout=0.6, lr=0.01, weight_decay=0.001, hidden_layers=[16]):
    runner = build_model(dataset, path, dropout, lr, weight_decay, hidden_layers, is_regression)
    results = []

    for train_p in train_size:
        res = execute_runner(runner, train_p=100, num_iter=1, early_stopping=True, val_p=1-train_p/100)
        tmp = list(res.values())[0]
        tmp['% train'] = train_p
        tmp['train'] = runner.loader.train_idx.cpu().numpy()
        results.append(tmp)
    runner.logger.info("Finished")

    return results


def main_clean(dataset, path=None, is_regression=False, train_size={70}, iterations=3, budget=0, early_stopping=False,
               out_intervals=20, val_p=0.1, dropout=0.6, lr=0.01, weight_decay=0.005, hidden_layers=[16]):

    runner = build_model(dataset, path, dropout, lr, weight_decay, hidden_layers, is_regression)
    # runner = build_model(dataset)

    # runner = ModelRunner(dataset, dataset_path, conf, logger=logger, data_logger=data_logger)

    # # train_p is the size wanted from each class, budget = train_p * number of classes
    # train_p = 20
    # results = execute_runner(runner, logger, train_p, num_iter=10)

    results = {}
    if budget > 0:
        if budget > 1:
            max_size = budget / runner.loader.data_size * 100
        else:
            max_size = budget * 100
        min_size = 0    # 5 / runner.loader.data_size * 100
        train_s = np.linspace(min_size, max_size, out_intervals)
    else:
        train_s = train_size
    for train_p in train_s:
        res = []
        for i in range(iterations):
            res.append(execute_runner(runner, train_p, num_iter=3, early_stopping=early_stopping, val_p=val_p))
        results[train_p] = mean_results(res)
    runner.logger.info("Finished")

    return results


if __name__ == "__main__":
    data_set = 'signaling_pathways'
    data_path = os.path.join("C:/roy/roy3/data_sets", data_set)
    main_clean(data_set, data_path)
