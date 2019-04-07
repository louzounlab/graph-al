from torch.utils.data import Dataset
import numpy as np
import torch
import copy


class DynamicData(Dataset):
    def __init__(self, components: dict, labels: dict, gpu=False):
        self._names = list(labels.keys())
        self._components = components
        self._labels = labels
        self._gpu = gpu

    def update(self, new_components: dict, new_labels: dict):
        new_names = list(new_labels.keys())
        self._names = list(set(self._names + new_names))
        for name in new_names:
            self._components[name] = new_components[name]
            self._labels[name] = new_labels[name] if name not in self._labels else self._labels[name]

    def __getitem__(self, item):
        # x = torch.Tensor(self._components[self._names[item]])
        rand = np.random.uniform(0, 1)
        if rand > 0.5:
            blacks = [name for name, val in self._labels.items() if val == 0]
            if len(blacks) != 0:
                np.random.shuffle(blacks)
                x = copy.deepcopy(torch.Tensor(self._components[blacks[0]]))
            else:
                x = copy.deepcopy(torch.Tensor(self._components[self._names[item]]))
        else:
            x = copy.deepcopy(torch.Tensor(self._components[self._names[item]]))

        label = torch.Tensor([self._labels[self._names[item]]])
        if self._gpu:
            x.cuda()
            label.cuda()
        return x, label

    def __len__(self):
        return len(self._names)


class DynamicDataManager:
    def __init__(self,  components: dict, labels: dict, train_size=0.8, gpu=False):
        self._train_size = train_size
        self._test_size = 1 - train_size
        self._test_names = []
        self._train_names = []

        train_x, train_y, test_x, test_y = self._split_components(components, labels)
        self.test = DynamicData(test_x, test_y, gpu=gpu)
        self.train = DynamicData(train_x, train_y, gpu=gpu)

    def update(self, new_components: dict, new_labels: dict):
        train_x, train_y, test_x, test_y = self._split_components(new_components, new_labels)
        self.test.update(test_x, test_y)
        self.train.update(train_x, train_y)

    def _split_names(self, new_name_list):
        free_names = []
        for name in new_name_list:
            if name not in self._test_names and name not in self._train_names:
                free_names.append(name)

        np.random.shuffle(free_names)
        split = int(len(free_names) * self._train_size)

        self._train_names += free_names[0:split]
        self._test_names += free_names[split: len(free_names)]

    def _split_components(self, components: dict, labels: dict):
        self._split_names(list(labels.keys()))
        train_x = {}
        train_y = {}
        test_x = {}
        test_y = {}

        for name, label in labels.items():
            if name in self._train_names:
                train_x[name] = components[name]
                train_y[name] = label
            else:
                test_x[name] = components[name]
                test_y[name] = label

        return train_x, train_y, test_x, test_y


