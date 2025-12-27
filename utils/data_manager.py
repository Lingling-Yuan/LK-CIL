import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import CXR14, CXR15, CCH5000, NIHCXR_LT, HAM10000, CheXpert

class DataManager:
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args):
        self.args = args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "Not enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    @property
    def nb_classes(self):
        return len(self._class_order)

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False, m_rate=None):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError(f"Unknown data source {source}.")

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose([*self._test_trsf,
                                       transforms.RandomHorizontalFlip(p=1.0),
                                       *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError(f"Unknown mode {mode}.")

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(x, y, idx, idx + 1)
            else:
                class_data, class_targets = self._select_rmm(x, y, idx, idx + 1, m_rate)
            data.append(class_data)
            targets.append(class_targets)

        if appendent:
            ad, at = appendent
            data.append(ad)
            targets.append(at)

        data = np.concatenate(data)
        targets = np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(self, indices, source, mode, appendent=None, val_samples_per_class=0):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError(f"Unknown data source {source}.")

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError(f"Unknown mode {mode}.")

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            cd, ct = self._select(x, y, idx, idx + 1)
            val_idx = np.random.choice(len(cd), val_samples_per_class, replace=False)
            train_idx = list(set(range(len(cd))) - set(val_idx))
            val_data.append(cd[val_idx]); val_targets.append(ct[val_idx])
            train_data.append(cd[train_idx]); train_targets.append(ct[train_idx])

        if appendent:
            ad, at = appendent
            for cls in range(int(at.max()) + 1):
                cd, ct = self._select(ad, at, cls, cls + 1)
                val_idx = np.random.choice(len(cd), val_samples_per_class, replace=False)
                train_idx = list(set(range(len(cd))) - set(val_idx))
                val_data.append(cd[val_idx]); val_targets.append(ct[val_idx])
                train_data.append(cd[train_idx]); train_targets.append(ct[train_idx])

        train_data = np.concatenate(train_data)
        train_targets = np.concatenate(train_targets)
        val_data = np.concatenate(val_data)
        val_targets = np.concatenate(val_targets)

        return (DummyDataset(train_data, train_targets, trsf, self.use_path),
                DummyDataset(val_data, val_targets, trsf, self.use_path))

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        if self.dataset_name.lower() in ("nihcxr_lt", "nihcxrlt") \
                or self._train_targets.ndim == 2:
            n_classes = self._train_targets.shape[1]
            self._class_order = list(range(n_classes))
            logging.info(f"Multi-label dataset, class_order = {self._class_order}")
            return

        labels = np.unique(self._train_targets)
        order = labels.tolist()
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(labels).tolist()
        self._class_order = order
        logging.info(f"Class order: {self._class_order}")

        self._train_targets = _map_new_class_index(self._train_targets, order)
        self._test_targets = _map_new_class_index(self._test_targets, order)

    def _select(self, x, y, low, high):
        if y.ndim == 1:
            idxs = np.where((y >= low) & (y < high))[0]
            return np.array(x)[idxs], np.array(y)[idxs]

        indexes = []
        for i in range(len(y)):
            pos_labels = np.where(y[i] == 1)[0]
            if len(pos_labels) == 0:
                continue
            if np.any((pos_labels >= low) & (pos_labels < high)):
                indexes.append(i)

        return np.array(x)[indexes], y[indexes]

    def _select_rmm(self, x, y, low, high, m_rate):
        idxs = np.where((y >= low) & (y < high))[0]
        if m_rate and m_rate != 0:
            sel = np.random.randint(0, len(idxs), int((1 - m_rate) * len(idxs)))
            idxs = np.sort(idxs[sel])
        return x[idxs], y[idxs]

    def getlen(self, index):
        return np.sum(self._train_targets == index)

class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB") if self.use_path \
              else Image.fromarray(self.images[idx])
        return idx, self.trsf(img), self.labels[idx]

def _map_new_class_index(y, order):
    return np.array([order.index(v) for v in y])



def _get_idata(name, args=None):
    n = name.lower()
    if n == "cxr14":
        return CXR14()
    elif n == "cxr15":
        return CXR15()
    elif n == "chexpert":
        return CheXpert()
    elif n == "cch5000":
        return CCH5000()
    elif n == "ham10000":
        return HAM10000()
    elif n in ("nihcxr_lt", "nihcxrlt"):
        return NIHCXR_LT()
    else:
        raise NotImplementedError(f"Unknown dataset {name}")


def pil_loader(path):
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")
