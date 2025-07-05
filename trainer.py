import sys
import logging
import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from utils.factory import get_model
from utils.data_manager import DataManager
from utils.toolkit import count_parameters

def train(args):
    return _train(args)

def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logdir = f"logs/{args['model_name']}/{args['dataset']}/{init_cls}/{args['increment']}"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{args['prefix']}_{args['seed']}_{args['backbone_type']}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    args["nb_classes"] = data_manager.nb_classes

    model = get_model(args["model_name"], args)
    device = args["device"][0]

    auc_matrix = []
    avg_auc = np.nan

    for task in range(data_manager.nb_tasks):
        logging.info(
            f"Total params: {count_parameters(model._network)}; "
            f"Trainable: {count_parameters(model._network, trainable=True)}"
        )
        model.incremental_train(data_manager)

        all_y_true, all_y_score = [], []
        model._network.eval()
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(model.test_loader):
                inputs = inputs.to(device)
                y = targets.flatten().cpu().numpy()
                logits = model._network.forward_orig(inputs)["logits"][:, :args["nb_classes"]]
                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_y_true.append(y)
                all_y_score.append(probs)

        y_true = np.concatenate(all_y_true)
        y_score = np.concatenate(all_y_score, axis=0)[:, : args["nb_classes"]]
        n_cls = y_score.shape[1]

        if n_cls == 2:
            y_score_i = y_score[:, 1]
            mask = ~np.isnan(y_score_i)
            yt, ys = y_true[mask], y_score_i[mask]
            if len(ys) < 2 or len(np.unique(yt)) < 2:
                per_class_auc = np.array([np.nan])
            else:
                per_class_auc = np.array([roc_auc_score(yt, ys)])
        else:
            y_true_bin = label_binarize(y_true, classes=np.arange(n_cls))
            per_class_auc = []
            for i in range(n_cls):
                yt = y_true_bin[:, i]
                ys = y_score[:, i]
                mask = ~np.isnan(ys)
                yt_f, ys_f = yt[mask], ys[mask]
                if len(ys_f) < 2 or len(np.unique(yt_f)) < 2:
                    per_class_auc.append(np.nan)
                else:
                    per_class_auc.append(roc_auc_score(yt_f, ys_f))
            per_class_auc = np.array(per_class_auc)

        avg_auc = float(np.nanmean(per_class_auc))
        logging.info(f"Task {task} average AUC: {avg_auc:.4f}")
        print(f"Task {task} average AUC: {avg_auc:.4f}")

        seen = data_manager.get_task_size(task)
        auc_matrix.append(per_class_auc[:seen].tolist())

        model.after_task()

    return {
        "avg_auc": avg_auc,
        "auc_matrix": auc_matrix,
    }

def _set_device(args):
    gpus = []
    for d in args["device"]:
        if isinstance(d, torch.device):
            gpus.append(d)
        elif isinstance(d, int):
            gpus.append(torch.device(f"cuda:{d}"))
        elif isinstance(d, str):
            if d.isdigit():
                gpus.append(torch.device(f"cuda:{d}"))
            elif d.startswith("cuda"):
                gpus.append(torch.device(d))
            else:
                gpus.append(torch.device("cpu"))
        else:
            gpus.append(torch.device("cpu"))
    args["device"] = gpus

def _set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args):
    for k, v in args.items():
        logging.info(f"{k}: {v}")
