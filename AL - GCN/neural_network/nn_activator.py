import torch
import copy
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch import nn
from neural_network.nn_dataset import DynamicDataManager
from neural_network.simple_data_set import SimpleDataSet
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


# class to train models
class FeedForwardNet:
    def __init__(self, model: nn.Module, train_size=0.85, class_weights=None, gpu=False):
        self._len_data = 0
        self._train_size = train_size
        # init models with current models
        self._model = model
        self._gpu = gpu and torch.cuda.is_available()
        if self._gpu:
            self._model.cuda()
        # empty list for every model - y axis for plotting loss by epochs
        self._test_loader = None
        self._train_loader = None
        self._train_validation = None
        self._data = None
        self._class_weights = torch.Tensor(class_weights) if class_weights is not None else None

    def set_data(self, components, labels):
        self._data = SimpleDataSet(components, labels, gpu=self._gpu)
        self._len_data = len(self._data)

        train_size = int(self._train_size * self._len_data)
        test_size = self._len_data - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(self._data, [train_size, test_size])

        self._train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self._train_validation = DataLoader(train_dataset, batch_size=1, shuffle=False)
        self._test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # train a model, input is the enum of the model type
    def train(self, total_epoch, early_stop=None, validation_rate=1, reset_optimizer_lr=False, stop_loss=False):
        if reset_optimizer_lr:
            self._model.set_optimizer(lr=reset_optimizer_lr)
        # EarlyStopping
        best_model = copy.deepcopy(self._model)
        best_loss = 1
        prev_loss = 0
        curr_loss = 0
        count_no_improvement = 0
        loss_res = []        # return loss

        for epoch_num in range(total_epoch):
            # set model to train mode
            # self._model.train()

            # calc number of iteration in current epoch
            for batch_index, (data, label) in enumerate(self._train_loader):
                # print progress
                self._model.optimizer.zero_grad()               # zero gradients
                output = self._model(data)                      # calc output of current model on the current batch
                loss = F.cross_entropy(output, label,
                                       weight=self._class_weights)  # define loss node ( negative log likelihood)
                loss.backward()                                 # back propagation
                self._model.optimizer.step()                    # update weights

            t = "Train"

            if stop_loss:
                curr_loss = self._validate(self._test_loader)
                if best_loss > curr_loss < 1:
                    best_loss = curr_loss
                    best_model = copy.deepcopy(self._model)

            # validate
            if validation_rate and epoch_num % validation_rate == 0:
                curr_loss = self._validate(self._train_validation)
                loss_res.append(curr_loss)
                # print(str(epoch_num) + "/" + str(total_epoch))
                # print(t + " --- validation results = loss:\t" + str(curr_loss))

            # EarlyStopping
            count_no_improvement = 0 if prev_loss > curr_loss else count_no_improvement + 1
            prev_loss = curr_loss
            if early_stop and count_no_improvement > early_stop:
                break

        if stop_loss and best_loss > 0:
            self._model = best_model
        return loss_res

    # validation function only the model and the data are important for input, the others are just for print
    def _validate(self, data_loader):
        loss_count = 0
        self._model.eval()

        # run though all examples in validation set (from input)
        for batch_index, (data, label) in enumerate(data_loader):
            output = self._model(data)                                          # calc output of the model
            loss_count += F.cross_entropy(output, label).item()          # sum total loss of all iteration

        loss = float(loss_count / len(data_loader.dataset))
        return loss

    # Test
    def test(self, train=False, print_loss=False):
        if train:
            loss = self._validate(self._train_validation)
        else:
            loss = self._validate(self._test_loader)
        if print_loss:
            print("Test --- validation results = loss:\t" + str(loss))
        return loss

    def predict(self, data, probs=False):
        data_tensor = torch.Tensor(data)
        if self._gpu:
            data_tensor = data_tensor.cuda()
        results = [self._model(vec) for vec in data_tensor]
        if probs:
            return results
        pred = [int(a.max(0)[1]) for a in results]
        return [self._data.true_label[i] for i in pred]


