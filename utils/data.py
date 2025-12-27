import numpy as np
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from utils.toolkit import split_images_labels

import pandas as pd
import os
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels


class iData:
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        return [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    return t

class NIHCXR_LT(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf  = build_transform(False, None)
    common_trsf = [
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]

    def download_data(self):
        root = "./CXR14"
        img_dir = os.path.join(root, "images")

        train_df = pd.read_csv(os.path.join(root, "miccai2023_nih-cxr-lt_labels_train.csv"))
        test_df = pd.read_csv(os.path.join(root, "miccai2023_nih-cxr-lt_labels_test.csv"))

        all_cols = list(train_df.columns)

        label_cols = [
            c for c in all_cols
            if c not in ["id", "subj_id", "No Finding"]
        ]
        print("[sanity] label_cols =", label_cols, "len =", len(label_cols))

        self.class_names = label_cols

        self.train_data = [os.path.join(img_dir, fn) for fn in train_df["id"].values]
        self.test_data = [os.path.join(img_dir, fn) for fn in test_df["id"].values]

        self.train_targets = train_df[label_cols].values.astype(np.float32)
        self.test_targets = test_df[label_cols].values.astype(np.float32)

        self.class_order = np.arange(len(self.class_names)).tolist()

class CXR14(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
    class_order = np.arange(8).tolist()

    def download_data(self):
        train_dir = "./CXR14/train/"
        test_dir = "./CXR14/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class CXR15(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]

    CLASS_NAMES_14 = [
    "0_Atelectasis", "1_Cardiomegaly", "2_Effusion", "3_Infiltration",
    "4_Mass", "5_Nodule", "6_Pneumonia", "7_Pneumothorax",
    "8_Consolidation", "9_Edema", "10_Emphysema", "11_Fibrosis",
    "12_PleuralThickening", "13_Hernia",]

    class_names = CLASS_NAMES_14 + ["14_COVID19"]
    class_order = np.arange(len(class_names)).tolist()

    def download_data(self):
        train_dir = "./CXR14addCOVID19/train/"
        test_dir = "./CXR14addCOVID19/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        idx2cls = {v: k for k, v in train_dset.class_to_idx.items()}
        name2fixed = {name: i for i, name in enumerate(self.class_names)}

        def remap(imgs):
            data, targets = split_images_labels(imgs)
            new_targets = []
            for t in targets:
                cls_name = idx2cls[int(t)]
                if cls_name not in name2fixed:
                    raise ValueError(f"[CXR14] unknown class folder: {cls_name}, "
                                     f"expected one of {self.class_names}")
                new_targets.append(name2fixed[cls_name])
            return data, np.array(new_targets, dtype=np.int64)

        self.train_data, self.train_targets = remap(train_dset.imgs)
        self.test_data, self.test_targets = remap(test_dset.imgs)

class CheXpert(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
    class_order = np.arange(8).tolist()

    def download_data(self):
        train_dir = "./CheXpert/train/"
        test_dir = "./CheXpert/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class CCH5000(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
    class_order = np.arange(8).tolist()

    def download_data(self):
        train_dir = "./CCH5000/train/"
        test_dir = "./CCH5000/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class HAM10000(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
    class_order = np.arange(8).tolist()

    def download_data(self):
        train_dir = "./HAM10000/train/"
        test_dir = "./HAM10000/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
