from torch.utils.data import Dataset
from collections import Counter
import torch


class SimpleDataSet(Dataset):
    def __init__(self, components, labels, gpu=False):
        classes = Counter(labels)
        self._n_classes = len(classes)
        self.true_label = [c for c in classes]
        self._matrix = torch.Tensor(components)
        self._labels = [self.true_label.index(l) for l in labels]
        self._gpu = gpu
        if self._gpu:
            self._matrix = self._matrix.cuda()
            self._labels = torch.LongTensor(self._labels).cuda()

    def __getitem__(self, index):
        return self._matrix[index], self._labels[index]

    def __len__(self):
        return len(self._labels)
