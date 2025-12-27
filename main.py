import sys
import logging
import os
import time
import copy
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from sklearn.preprocessing import label_binarize
from utils.factory import get_model
from utils.data_manager import DataManager
from utils.toolkit import count_parameters

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

def train(args):
    return _train(args)


class CILEvaluator:

    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.acc_matrix = np.zeros((num_tasks, num_tasks))
        self.auc_matrix = np.zeros((num_tasks, num_tasks))

    def update(self, current_task_id, task_scores_acc, task_scores_auc):
        for t_id, score in enumerate(task_scores_acc):
            self.acc_matrix[current_task_id, t_id] = score
        for t_id, score in enumerate(task_scores_auc):
            self.auc_matrix[current_task_id, t_id] = score

    def compute_forgetting(self, current_task_id):
        if current_task_id == 0:
            return 0.0
        forgetting = []
        for j in range(current_task_id):
            max_perf = np.max(self.auc_matrix[:current_task_id, j]) 
            curr_perf = self.auc_matrix[current_task_id, j]         
            forgetting.append(max_perf - curr_perf)
        return np.mean(forgetting)

    def compute_bwt(self, current_task_id):
        if current_task_id == 0:
            return 0.0
        bwt = []
        for j in range(current_task_id):
            curr_perf = self.auc_matrix[current_task_id, j]
            orig_perf = self.auc_matrix[j, j]  
            bwt.append(curr_perf - orig_perf)
        return np.mean(bwt)


def compute_flops(model, device):
    if not THOP_AVAILABLE:
        return "N/A"
    try:
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        model_eval = copy.deepcopy(model)
        model_eval.eval()
        flops, params = profile(model_eval, inputs=(dummy_input,), verbose=False)
        return f"{flops / 1e9:.2f} G"
    except Exception:
        return "Error"


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

    evaluator = CILEvaluator(data_manager.nb_tasks)
    flops_str = compute_flops(model._network.backbone, device)
    auc_matrix_list_for_main = []

    avg_auc = np.nan
    avg_acc = np.nan
    mAP = 0.0
    forget_auc = 0.0
    bwt_auc = 0.0

    covid_auc = np.nan
    covid_acc = np.nan
    covid_idx = args.get("covid_class_idx", data_manager.nb_classes - 1)
    multi_label = args.get("multi_label", False)
    for task in range(data_manager.nb_tasks):
        total_params = count_parameters(model._network)
        trainable_params = count_parameters(model._network, trainable=True)
        logging.info(
            f"Task {task} | Params: {total_params} "
            f"(Trainable: {trainable_params}) | FLOPs: {flops_str}"
        )

        model.incremental_train(data_manager)
        model._network.eval()

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start_time = time.time()

        y_score, y_true = model._eval_cnn(model.test_loader, return_logits=False)
        num_seen = y_score.shape[1]
        y_score = y_score[:, :num_seen]
        if isinstance(y_true, np.ndarray) and y_true.ndim == 2:
            y_true = y_true[:, :num_seen]

        n_cls = num_seen

        if multi_label:
            if y_true.ndim == 1:
                raise ValueError(
                )

            per_class_auc = []
            per_class_acc = []

            for i in range(n_cls):
                yt = y_true[:, i].astype(np.float32)   # 0/1
                ys = y_score[:, i].astype(np.float32)  # [0,1]

                if yt.sum() == 0 or yt.sum() == len(yt):
                    per_class_auc.append(np.nan)
                else:
                    per_class_auc.append(roc_auc_score(yt, ys))

                y_pred_i = (ys >= 0.5).astype(int)
                per_class_acc.append(accuracy_score(yt, y_pred_i))

            per_class_auc = np.array(per_class_auc)
            per_class_acc = np.array(per_class_acc)

            avg_auc = float(np.nanmean(per_class_auc))
            avg_acc = float(np.mean(per_class_acc))

            aps = []
            for i in range(n_cls):
                yt = y_true[:, i]
                ys = y_score[:, i]
                if yt.sum() == 0:
                    continue
                aps.append(average_precision_score(yt, ys))
            mAP = float(np.mean(aps)) if len(aps) > 0 else 0.0

        else:
            if y_true.ndim == 2:
                y_true_1d = y_true.argmax(axis=1).astype(int)  # (N,)
            else:
                y_true_1d = y_true.astype(int)                 # (N,)

            y_score_2d = y_score  # (N, C_seen)

            per_class_auc = []
            per_class_acc = []
            y_pred = np.argmax(y_score_2d, axis=1)

            if n_cls == 2:
                y_score_pos = y_score_2d[:, 1]
                mask = ~np.isnan(y_score_pos)
                yt, ys = y_true_1d[mask], y_score_pos[mask]
                if len(ys) < 2 or len(np.unique(yt)) < 2:
                    per_class_auc = np.array([np.nan])
                else:
                    per_class_auc = np.array([roc_auc_score(yt, ys)])
                per_class_acc = np.array([accuracy_score(y_true_1d, y_pred)])
            else:
                y_true_bin = label_binarize(y_true_1d, classes=np.arange(n_cls))
                for i in range(n_cls):
                    yt = y_true_bin[:, i]
                    ys = y_score_2d[:, i]
                    mask = ~np.isnan(ys)
                    yt_f, ys_f = yt[mask], ys[mask]

                    if len(ys_f) < 2 or len(np.unique(yt_f)) < 2:
                        per_class_auc.append(np.nan)
                    else:
                        per_class_auc.append(roc_auc_score(yt_f, ys_f))

                    cls_idx = np.where(y_true_1d == i)[0]
                    if len(cls_idx) > 0:
                        cls_acc = accuracy_score(y_true_1d[cls_idx], y_pred[cls_idx])
                        per_class_acc.append(cls_acc)
                    else:
                        per_class_acc.append(0.0)

                per_class_auc = np.array(per_class_auc)
                per_class_acc = np.array(per_class_acc)

            avg_auc = float(np.nanmean(per_class_auc))
            avg_acc = float(np.mean(per_class_acc))

            try:
                if n_cls > 2:
                    y_true_bin_map = label_binarize(y_true_1d, classes=np.arange(n_cls))
                    mAP = average_precision_score(y_true_bin_map, y_score_2d, average="macro")
                else:
                    mAP = average_precision_score(y_true_1d, y_score_2d[:, 1])
            except Exception:
                mAP = 0.0

        covid_seen_now = (covid_idx < n_cls)
        if covid_seen_now:
            covid_auc = float(per_class_auc[covid_idx]) if covid_idx < len(per_class_auc) else np.nan
            covid_acc = float(per_class_acc[covid_idx]) if covid_idx < len(per_class_acc) else np.nan

        end_time = time.time()
        inference_time = end_time - start_time
        if torch.cuda.is_available() and device.type == "cuda":
            max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        else:
            max_mem = 0.0

        task_auc_scores = []
        task_acc_scores = []

        cursor = 0
        for t_idx in range(task + 1):
            t_size = data_manager.get_task_size(t_idx)

            t_aucs = per_class_auc[cursor: cursor + t_size]
            t_accs = per_class_acc[cursor: cursor + t_size]

            if len(t_aucs) == 0:
                task_auc_scores.append(0.0)
                task_acc_scores.append(0.0)
            else:
                if np.isnan(t_aucs).all():
                    task_auc_scores.append(0.0)
                else:
                    task_auc_scores.append(float(np.nanmean(t_aucs)))
                task_acc_scores.append(float(np.mean(t_accs)))

            cursor += t_size

        evaluator.update(task, task_acc_scores, task_auc_scores)
        forget_auc = evaluator.compute_forgetting(task)
        bwt_auc = evaluator.compute_bwt(task)

        logging.info("-" * 60)
        logging.info(f"Task {task} Evaluation Summary")
        logging.info("-" * 60)
        logging.info(f"Avg AUC (Global) : {avg_auc:.4f}")
        logging.info(f"Avg ACC (Global) : {avg_acc:.4f}")
        logging.info(f"mAP              : {mAP:.4f}")
        logging.info(f"Forgetting (AUC) : {forget_auc:.4f}")
        logging.info(f"BWT (AUC)        : {bwt_auc:.4f}")

        if covid_seen_now:
            logging.info(f"COVID19 (class_idx={covid_idx}) AUC : {covid_auc:.4f}")
            logging.info(f"COVID19 (class_idx={covid_idx}) ACC : {covid_acc:.4f}")
        else:
            logging.info(f"COVID19 (class_idx={covid_idx}) AUC/ACC : N/A (not seen yet)")

        logging.info("-" * 60)
        logging.info(f"Inference Time   : {inference_time:.2f} s")
        logging.info(f"Peak GPU Mem     : {max_mem:.2f} GB")
        logging.info("-" * 60)

        if covid_seen_now:
            print(
                f"Task {task} | Avg AUC: {avg_auc:.4f} | COVID19 AUC: {covid_auc:.4f} "
                f"| F-AUC: {forget_auc:.4f} | Time: {inference_time:.1f}s"
            )
        else:
            print(
                f"Task {task} | Avg AUC: {avg_auc:.4f} "
                f"| F-AUC: {forget_auc:.4f} | Time: {inference_time:.1f}s"
            )

        auc_matrix_list_for_main.append(per_class_auc.tolist())
        model.after_task()

    forget_acc = 0.0
    bwt_acc = 0.0

    return {
        "avg_auc": avg_auc,
        "avg_acc": avg_acc,
        "map": mAP,
        "forget_auc": forget_auc,
        "forget_acc": forget_acc,
        "bwt_auc": bwt_auc,
        "bwt_acc": bwt_acc,
        "covid_auc": covid_auc,
        "covid_acc": covid_acc,
        "auc_matrix": auc_matrix_list_for_main,
        "evaluator_auc_matrix": evaluator.auc_matrix.tolist(),
        "evaluator_acc_matrix": evaluator.acc_matrix.tolist(),
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
