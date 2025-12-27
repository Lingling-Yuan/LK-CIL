import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import LKCILNet  # Language-guided CIL network with adapters and cosine head
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, target2onehot
from torch.distributions.multivariate_normal import MultivariateNormal
from utils.text_feature_manager import TextFeatureManager
import copy

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        # (1) Initialize the LKCIL network: backbone + adapters + cosine classification head
        self._network = LKCILNet(args, True)
        # (2) Copy current adapters as global reference for future merging
        self._network.backbone.global_adapter = copy.deepcopy(self._network.backbone.cur_adapter)

        # Text manager to load prompts and text prototypes for each class
        self.txt_mgr = TextFeatureManager(
            prompts_path=args['prompt_json_path'],
            text_feats_path=args['text_feats_path'],
            keys_path=args['text_feats_keys_path'],
            device=self._device
        )

        # Hyperparameters: only alpha is retained for inference blending
        self.alpha = args.get('alpha', 0.5)         # Fusion coefficient α for ensemble blending
        self.lambda_orth = args.get('lambda_orth', 0.3)  # Weight λ_orth for orthogonality loss
        self.lambda_align = args.get('lambda_align', 0.3) # Weight λ_align for alignment loss

        D = self._network.backbone.out_dim          # Dimensionality of backbone features
        hidden_dim = args.get("graph_hidden_dim", 128)  # Hidden size for prototype refinement GCN

        # 1) Projection layer: concatenated visual+text prototype (2D) → D
        self.proto_proj = nn.Linear(2 * D, D).to(self._device)

        # 2) Graph selector: two-layer MLP for prototype refinement (GCN-like)
        self.graph_selector = nn.Sequential(
            nn.Linear(D, hidden_dim),  # D → hidden
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, D),  # hidden → D
        ).to(self._device)

        # Containers for class means, covariances, and mapping from class to task ID
        self.cls_mean = dict()
        self.cls_cov = dict()
        self.cls2task = dict()

        # Training settings
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.ca_lr = args["ca_lr"]  # LR for classifier alignment stage
        self.crct_epochs = args["crct_epochs"]  # Number of correction epochs for CA
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.ensemble = args.get("ensemble", False)  # Whether to ensemble during inference

        # Freeze all backbone parameters except adapters and head
        for n, p in self._network.backbone.named_parameters():
            if 'adapter' not in n and 'head' not in n:
                p.requires_grad = False

        # Log total vs trainable parameter counts
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        # If any parameters remain trainable, list them
        if total_params != total_trainable_params:
            for name, param in self._network.backbone.named_parameters():
                if param.requires_grad:
                    logging.info(f"{name}: {param.numel()}")

    def replace_fc(self):
        """
        After adapter fine-tuning, replace the classifier weights
        with class prototypes (mean of features) for stability.
        """
        model = self._network.to(self._device)
        embedding_list = []
        label_list = []

        with torch.no_grad():
            for _, batch in enumerate(self.train_loader_for_protonet):
                (_, data, label) = batch
                data = data.to(self._device)
                embedding = model.forward_orig(data)["features"]  # (B, D)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        embedding_list = torch.cat(embedding_list, dim=0)  # (N, D) on CPU
        label_list = torch.cat(label_list, dim=0)  

        weight_device = self._network.fc.weight.device
        embedding_list = embedding_list.to(weight_device)
        label_list = label_list.to(weight_device)

        # ---------- Single-label ----------
        if label_list.ndim == 1:
            class_ids = torch.unique(label_list).tolist()
            for class_index in class_ids:
                class_index = int(class_index)
                idx = (label_list == class_index).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                e = embedding_list[idx]  # (n_c, D)
                proto = e.mean(0)
                self._network.fc.weight.data[class_index].copy_(proto)

        # ---------- Multi-label ----------
        elif label_list.ndim == 2:
            num_classes = label_list.size(1)

            for c in range(self._known_classes, self._total_classes):
                if c >= num_classes:
                    break  

                mask = label_list[:, c] > 0.5
                if mask.sum() == 0:
                    continue
                e = embedding_list[mask]  # (n_c, D)
                proto = e.mean(0)
                self._network.fc.weight.data[c].copy_(proto)

        else:
            raise ValueError(f"Unexpected label dimension: {label_list.shape}")

        return model

    def after_task(self):
        # Update known classes after finishing a task
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        """
        Main loop for incremental training of each new task.
        """
        self.data_manager = data_manager

        # Reset current adapter to global state before fine-tuning
        self._network.backbone.cur_adapter = copy.deepcopy(self._network.backbone.global_adapter)
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        all_ids = sorted(
            list(self.txt_mgr.prompts.keys()),
            key=lambda k: int(k.split('_')[0])
        )
        assert len(all_ids) >= self._total_classes
        self.class_keys = all_ids[:self._total_classes]
        self.class_proto_tensor = self.txt_mgr.get_class_proto_tensor(self.class_keys)

        for i in range(self._known_classes, self._total_classes):
            self.cls2task[i] = self._cur_task

        self._network.update_fc(self._total_classes)
        logging.info(f"Learning on {self._known_classes}-{self._total_classes}")

        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test",
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        for idx, class_id in enumerate(self.class_keys):
            print(f"[sanity] label {idx} ⇒ class_id '{class_id}'")

        first_batch = next(iter(self.train_loader))
        _, inputs, targets = first_batch

        if isinstance(targets, torch.Tensor):
            if targets.ndim == 1:
                print(f"[sanity] sample targets: {targets[:10].tolist()}")
                try:
                    mapped = [self.class_keys[int(t)] for t in targets[:10].cpu().tolist()]
                    print(f"[sanity] sample class_ids: {mapped}")
                except Exception:
                    pass
            elif targets.ndim == 2:
                print(f"[sanity] sample multi-label targets (first 2): {targets[:2].tolist()}")
        else:
            print(f"[sanity] sample targets (non-tensor): {targets[:10]}")

        if len(self._multiple_gpus) > 1:
            print("Multiple GPUs")
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)
        self.replace_fc()

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        """
        Adapter fine-tuning stage: update only adapter modules and classification head.
        """
        self._network.backbone.to(self._device)
        optimizer = self.get_optimizer(self._network.backbone)
        scheduler = self.get_scheduler(optimizer)
        self._init_train(train_loader, test_loader, optimizer, scheduler)

        # After initial tuning, allow adapter_update (if supported)
        self._network.backbone.adapter_update()

        # Dynamic Language-guided Knowledge Merge
        new_cids = self.class_keys[self._known_classes:self._total_classes]
        K_i = self.txt_mgr.update_task_proto(new_cids)  # Task-specific prototypes
        alpha_i = self.txt_mgr.compute_merge_alpha(K_i)
        print(alpha_i)
        A_i = self._network.backbone.adapter_list[-1]  # Current task adapter
        global_adapter = self._network.backbone.global_adapter

        # Merge layer weights: A_g ← (1-α) A_g + α A_i
        for g_layer, a_layer in zip(global_adapter, A_i):
            g_layer.down_proj.weight.data.mul_(1 - alpha_i) \
                .add_(a_layer.down_proj.weight.data, alpha=alpha_i)
            g_layer.down_proj.bias.data.mul_(1 - alpha_i) \
                .add_(a_layer.down_proj.bias.data, alpha=alpha_i)
            g_layer.up_proj.weight.data.mul_(1 - alpha_i) \
                .add_(a_layer.up_proj.weight.data, alpha=alpha_i)
            g_layer.up_proj.bias.data.mul_(1 - alpha_i) \
                .add_(a_layer.up_proj.bias.data, alpha=alpha_i)

        # Compute class prototype statistics (means, covariances)
        self._compute_mean(self._network.backbone)
        # If not the first task, run classifier alignment
        if self._cur_task > 0:
            self.classifer_align(self._network.backbone)

    def get_optimizer(self, model):
        """Build optimizer: higher LR for adapters, lower for fc/head layers."""
        base_params = [{'params': [p for n, p in model.named_parameters()
                                    if 'adapter' in n and p.requires_grad],
                        'lr': self.init_lr, 'weight_decay': self.weight_decay}]
        base_fc_params = [{'params': [p for n, p in model.named_parameters()
                                      if 'adapter' not in n and p.requires_grad],
                            'lr': self.init_lr * 0.1, 'weight_decay': self.weight_decay}]
        network_params = base_params + base_fc_params
        if self.args['optimizer'] == 'sgd':
            return optim.SGD(network_params, momentum=0.9)
        elif self.args['optimizer'] == 'adam':
            return optim.Adam(network_params)
        elif self.args['optimizer'] == 'adamw':
            return optim.AdamW(network_params)

    def get_scheduler(self, optimizer):
        """LR scheduler selection."""
        if self.args['scheduler'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        elif self.args['scheduler'] == 'steplr':
            return optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.args['init_milestones'], gamma=self.args['init_lr_decay'])
        elif self.args['scheduler'] == 'constant':
            return None

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        """
        Core training loop for adapter fine-tuning, including alignment and orthogonality losses.
        """
        prog_bar = tqdm(range(self.args["tuned_epoch"]))

        for epoch in prog_bar:
            self._network.backbone.train()
            losses = 0.0
            correct = torch.tensor(0.0, device=self._device)
            total = 0

            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device)

                if targets.dim() == 1:
                    targets = targets.to(self._device).long()

                    output = self._network(
                        inputs, adapter_id=self._cur_task, train=True
                    )
                    logits = output["logits"][:, : self._total_classes]

                    if self._known_classes > 0:
                        logits[:, : self._known_classes] = float("-inf")

                    ce = F.cross_entropy(logits, targets)

                    orth = self.orth_loss(output["pre_logits"], targets)

                    v = output["pre_logits"]  # (B, D)
                    text_proto = self.class_proto_tensor[targets]  # (B, D)
                    cos_sim = F.cosine_similarity(v, text_proto, dim=1)
                    align = (1.0 - cos_sim).mean()

                    loss = ce + orth + self.lambda_align * align

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    losses += loss.item()

                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets).float().sum()
                    total += targets.size(0)

                else:  # targets.dim() == 2
                    targets = targets.to(self._device).float()

                    output = self._network(
                        inputs, adapter_id=self._cur_task, train=True
                    )
                    logits = output["logits"]  # (B, C_all)

                    logits_task = logits[:, : self._total_classes]
                    targets_task = targets[:, : self._total_classes]

                    bce = F.binary_cross_entropy_with_logits(
                        logits_task, targets_task
                    )

                    orth = self.orth_loss(output["pre_logits"], targets_task)

                    v = output["pre_logits"]  # (B, D)
                    text_proto_seen = self.class_proto_tensor[
                                      : self._total_classes, :
                                      ]  # (C_seen, D)

                    pos_counts = targets_task.sum(dim=1, keepdim=True)  # (B, 1)

                    norm_weights = torch.where(
                        pos_counts > 0,
                        targets_task / (pos_counts + 1e-8),
                        torch.zeros_like(targets_task),
                    )  # (B, C_seen)

                    # (B, C_seen) @ (C_seen, D) -> (B, D)
                    sample_text_proto = norm_weights @ text_proto_seen

                    valid_mask = pos_counts.squeeze(1) > 0
                    if valid_mask.any():
                        cos_sim = F.cosine_similarity(
                            v[valid_mask], sample_text_proto[valid_mask], dim=1
                        )
                        align = (1.0 - cos_sim).mean()
                    else:
                        align = torch.tensor(0.0, device=self._device)

                    loss = bce + orth + self.lambda_align * align

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    losses += loss.item()

                    with torch.no_grad():
                        probs = torch.sigmoid(logits_task)
                        preds = (probs > 0.5).float()
                        correct += (preds == targets_task).float().sum()
                        total += targets_task.numel()

            if scheduler:
                scheduler.step()

            train_acc = np.around(
                tensor2numpy(correct) * 100.0 / (total + 1e-8),
                decimals=2,
            )
            prog_bar.set_description(
                f"Task {self._cur_task}, Epoch {epoch + 1}/{self.args['tuned_epoch']} => "
                f"Loss {losses / max(1, len(train_loader)):.3f}, TrainAccy {train_acc:.2f}"
            )

        logging.info(prog_bar)

    @torch.no_grad()
    def _compute_mean(self, model):
        """
        Compute visual feature means and covariances (or variances) per class.
        Then concatenate with text prototypes and project to form unified prototypes.
        """
        model.eval()
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset,
                batch_size=self.batch_size * 3,
                shuffle=False,
                num_workers=4,
            )
            vectors = []
            for _, _inputs, _ in idx_loader:
                feats = model(
                    _inputs.to(self._device),
                    adapter_id=self._cur_task,
                    train=True,
                )["features"]
                vectors.append(feats)
            vectors = torch.cat(vectors, dim=0)

            if self.args["ca_storage_efficient_method"] in ["covariance", "variance"]:
                self.cls_mean[class_idx] = vectors.mean(dim=0).to(self._device)
                cov = torch.cov(vectors.T) + torch.eye(
                    vectors.size(1), device=self._device
                ) * 1e-4
                self.cls_cov[class_idx] = (
                    cov
                    if self.args["ca_storage_efficient_method"] == "covariance"
                    else torch.diag(cov)
                )
            else:
                from sklearn.cluster import KMeans

                arr = vectors.cpu().numpy()
                kmeans = KMeans(
                    n_clusters=self.args["n_centroids"], n_init="auto"
                ).fit(arr)
                means, vars_ = [], []
                for i in range(self.args["n_centroids"]):
                    cluster = arr[kmeans.labels_ == i]
                    means.append(torch.tensor(cluster.mean(0), device=self._device))
                    vars_.append(torch.tensor(cluster.var(0), device=self._device))
                self.cls_mean[class_idx], self.cls_cov[class_idx] = means, vars_

        # Stack visual means and combine with text prototypes
        vis_proto = torch.stack(
            [self.cls_mean[c] for c in range(self._total_classes)], dim=0
        )  # (C, D)
        text_proto = self.class_proto_tensor  # (C, D)
        proto_2d = torch.cat([vis_proto, text_proto], dim=1)  # (C, 2D)
        proto_d = self.proto_proj(proto_2d)
        # Normalize to get final class prototypes
        self.cat_proto = F.normalize(proto_d, p=2, dim=1)  # (C, D)

    def classifer_align(self, model):
        """
        Classifier alignment: generate synthetic samples via multivariate normal
        and fine-tune classifier head to correct bias across classes.
        """
        model.train()
        param_list = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and "adapter" not in n
        ]
        optimizer = optim.SGD(
            [{"params": param_list, "lr": self.ca_lr, "weight_decay": self.weight_decay}],
            momentum=0.9,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.crct_epochs
        )
        prog_bar = tqdm(range(self.crct_epochs))
        num_sampled = self.batch_size * 5

        for epoch in prog_bar:
            sampled_data, sampled_label = [], []

            if self.args["ca_storage_efficient_method"] in ["covariance", "variance"]:
                for class_idx in range(self._total_classes):
                    mean, cov = self.cls_mean[class_idx], self.cls_cov[class_idx]
                    m = MultivariateNormal(mean.float(), cov.float())
                    sampled = m.sample((num_sampled,))  # synthetic draws
                    sampled_data.append(sampled)
                    sampled_label.extend([class_idx] * num_sampled)
            else:
                raise NotImplementedError

            inputs = torch.cat(sampled_data).to(self._device)
            targets = torch.tensor(sampled_label).long().to(self._device)
            sf = torch.randperm(inputs.size(0))
            inputs, targets = inputs[sf], targets[sf]

            losses, correct, total = 0.0, 0, 0
            for i in range(self._total_classes):
                inp = inputs[i * num_sampled: (i + 1) * num_sampled]
                tgt = targets[i * num_sampled: (i + 1) * num_sampled]
                outputs = model(inp, fc_only=True)["logits"][:, : self._total_classes]
                loss = F.cross_entropy(outputs, tgt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, dim=1)
                correct += preds.eq(tgt).cpu().sum()
                total += len(tgt)
                losses += loss.item()

            scheduler.step()
            acc = np.round(tensor2numpy(correct) * 100 / total, 2)
            prog_bar.set_description(
                f"Task {self._cur_task}, CA Epoch {epoch + 1}/{self.crct_epochs} => "
                f"Loss {losses / self._total_classes:.3f}, CAAcc {acc:.2f}"
            )
        logging.info(prog_bar)

    def orth_loss(self, features, targets):
        """
        Orthogonality loss: encourage feature diversity by pushing apart samples.
        If class means available, incorporate them for extra regularization.
        """
        if self.cls_mean:
            # Stack class means and current features
            sample_mean = torch.stack(
                [v for v in self.cls_mean.values()]
                if not isinstance(list(self.cls_mean.values())[0], list)
                else [m for means in self.cls_mean.values() for m in means],
                dim=0,
            ).to(self._device)
            M = torch.cat([sample_mean, features], dim=0)
            sim = torch.matmul(M, M.t()) / 0.8
            loss = F.cross_entropy(
                sim, torch.arange(sim.size(0)).long().to(self._device)
            )
            return self.lambda_orth * loss
        else:
            sim = torch.matmul(features, features.t()) / 0.8
            loss = F.cross_entropy(
                sim, torch.arange(sim.size(0)).long().to(self._device)
            )
            return self.lambda_orth * loss

    @torch.no_grad()
    def _eval_cnn(self, loader, return_logits=False):
        """
        Evaluation with graph-based prototype refinement and optional ensemble.
        """
        import os
        import time
        import hashlib

        from utils.debug_eval import (
            EvalDebugConfig,
            JsonlWriter,
            export_prototype_graph,
            compute_correct_single_label,
            compute_correct_multi_label,
        )
        from utils.viz_cam import (
            tensor_to_pil,
            attention_rollout,
            vit_gradcam_last_attn,
            patch_text_cosine_heatmap,
            save_overlay,
        )

        # --- try import save_raw_image; fallback to PIL save if missing ---
        try:
            from utils.viz_cam import save_raw_image
            _HAS_SAVE_RAW = True
        except Exception:
            _HAS_SAVE_RAW = False

        # Reset adapter to merged global state
        try:
            if hasattr(self, "global_adapter"):
                self._network.backbone.cur_adapter = self.global_adapter
            elif hasattr(self._network.backbone, "global_adapter"):
                self._network.backbone.cur_adapter = self._network.backbone.global_adapter
        except Exception:
            pass

        self._network.eval()

        y_pred_list, y_true_list, orig_score_list = [], [], []
        cos = nn.CosineSimilarity(dim=-1)
        MAX_ITER = int(self.args.get("eval_max_iter", 4))
        C = int(self._total_classes)

        # -------- timing configs --------
        timing_sync_cuda = bool(self.args.get("timing_sync_cuda", True))

        def _sync_if_needed(t: torch.Tensor):
            if timing_sync_cuda and torch.cuda.is_available() and t.is_cuda:
                torch.cuda.synchronize()

        per_image_time_ms_all = []

        # -------- debug / export configs --------
        debug_cfg = EvalDebugConfig(
            enabled=bool(self.args.get("eval_debug", False)),
            out_dir=str(self.args.get("eval_debug_dir", "./debug")),
            jsonl_name=str(self.args.get("eval_debug_jsonl", "eval_routing_log.jsonl")),
            only_errors=bool(self.args.get("eval_debug_only_errors", True)),
            max_samples=int(self.args.get("eval_debug_max_samples", 2000)),
            log_topk=int(self.args.get("eval_debug_topk", 5)),
        )

        export_graph = bool(self.args.get("export_proto_graph", False))
        export_graph_csv = bool(self.args.get("export_proto_graph_csv", True))
        export_graph_json = bool(self.args.get("export_proto_graph_json", True))

        viz_enable = bool(self.args.get("viz_enable", False))
        viz_only_errors = bool(self.args.get("viz_only_errors", True))
        viz_max_samples = int(self.args.get("viz_max_samples", 100))
        viz_alpha = float(self.args.get("viz_alpha", 0.45))
        viz_dpi = int(self.args.get("viz_dpi", 200))
        viz_use_rollout = bool(self.args.get("viz_attn_rollout", True))
        viz_use_gradcam = bool(self.args.get("viz_vit_gradcam", False))
        viz_use_patch_text = bool(self.args.get("viz_patch_text", True))
        viz_patch_target = str(self.args.get("viz_patch_target", "pred"))  # "pred" or "true"

        debug_writer = None
        debug_written = 0
        viz_written = 0
        graph_exported = False

        # -------- prompts version tag (md5) --------
        def _file_md5(path: str) -> str:
            try:
                if (not path) or (not os.path.exists(path)) or (not os.path.isfile(path)):
                    return "missing"
                h = hashlib.md5()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                return h.hexdigest()
            except Exception:
                return "unknown"

        prompt_json_path = (
                str(self.args.get("prompt_json", "")) or
                str(self.args.get("prompt_path", "")) or
                str(self.args.get("prompt_file", ""))
        )
        prompt_md5 = _file_md5(prompt_json_path) if prompt_json_path else "no_prompt"
        prompt_tag = f"prompt_{prompt_md5[:8]}"

        # output dirs
        if debug_cfg.enabled or export_graph or viz_enable:
            task_dir = os.path.join(debug_cfg.out_dir, f"task_{self._cur_task}")
            os.makedirs(task_dir, exist_ok=True)
            run_dir = os.path.join(task_dir, prompt_tag)
            os.makedirs(run_dir, exist_ok=True)
        else:
            task_dir = None
            run_dir = None

        # debug jsonl writer
        if debug_cfg.enabled and run_dir is not None:
            debug_path = os.path.join(run_dir, debug_cfg.jsonl_name)

            # overwrite optional (avoid appending across runs)
            if bool(self.args.get("eval_debug_overwrite", True)) and os.path.exists(debug_path):
                try:
                    os.remove(debug_path)
                except Exception:
                    pass

            debug_writer = JsonlWriter(debug_path)
            debug_writer.__enter__()

        # export graph buffers (to avoid including IO in timing)
        graph_export_buffer = None  # (sim, idx, vals, A, tau)

        for batch_idx, (indices, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            # -------- detect single-label vs multi-label --------
            is_single_label = (targets.dim() == 1)  # (B,)
            if is_single_label:
                targets_seen = targets.long()  # (B,)
            else:
                targets_seen = targets.float()[:, :C]  # (B, C)

            # ========== START timing (inference only, exclude debug/viz I/O) ==========
            _sync_if_needed(inputs)
            t0 = time.perf_counter()

            # -------- original/global adapter prediction (z^(0)) --------
            orig_logits = self._network.forward_orig(inputs)["logits"][:, :C]  # (B, C)
            if is_single_label:
                orig_prob_for_score = F.softmax(orig_logits, dim=1)
            else:
                orig_prob_for_score = torch.sigmoid(orig_logits)
            orig_score_list.append(orig_prob_for_score.detach().cpu().numpy())

            # -------- features from each task adapter --------
            B = inputs.size(0)
            T = int(self._cur_task + 1)
            D = int(self._network.backbone.out_dim)
            all_features = torch.zeros(B, T, D, device=self._device)

            for t in range(T):
                all_features[:, t, :] = self._network.backbone(
                    inputs, adapter_id=t, train=False
                )["features"]  # (B, D)

            # -------- graph-based prototype refinement --------
            X = self.cat_proto[:C].to(self._device)  # (C, D)
            X = F.normalize(X, dim=1)

            sim = cos(
                X.unsqueeze(1).expand(-1, C, -1),
                X.unsqueeze(0).expand(C, -1, -1),
            )  # (C, C)

            K = min(int(self.args.get("graph_sparsity_k", 5)), C)
            tau = float(self.args.get("graph_tau", 1.0))

            mask = torch.full_like(sim, float("-inf"))
            vals, idx = sim.topk(K, dim=1)
            mask.scatter_(1, idx, vals)
            A = F.softmax(mask / tau, dim=1)  # (C, C)

            # buffer graph export (do IO after timing)
            if export_graph and (not graph_exported) and run_dir is not None and graph_export_buffer is None:
                graph_export_buffer = (sim.detach(), idx.detach(), vals.detach(), A.detach(), float(tau))

            # Two-layer "GCN-like" refinement using graph_selector weights
            W1, b1 = self.graph_selector[0].weight, self.graph_selector[0].bias
            H1 = F.relu(A @ X @ W1.t() + b1)
            W2, b2 = self.graph_selector[2].weight, self.graph_selector[2].bias
            H = A @ H1 @ W2.t() + b2  # (C, D)
            H = F.normalize(H, dim=1)

            # -------- initial adapter selection (t*) --------
            sims = torch.zeros(B, T, C, device=self._device)
            for t in range(T):
                sims[:, t, :] = cos(
                    all_features[:, t, :].unsqueeze(1).expand(-1, C, -1),
                    H.unsqueeze(0).expand(B, -1, -1),
                )

            sims_flat = sims.view(B, -1)  # (B, T*C)
            best_flat = sims_flat.argmax(dim=1)  # (B,)
            best_t = (best_flat // C).long()  # (B,)
            best_c = (best_flat % C).long()  # (B,)
            refined_idx = best_t  # (B,)

            # -------- self-refinement loop --------
            final_logits_list = []
            routing_traces = [None for _ in range(B)]

            for b in range(B):
                prev = int(refined_idx[b].item())
                logits_b = None

                trace = {
                    "init_best_t": int(best_t[b].item()),
                    "init_best_c": int(best_c[b].item()),
                    "iters": [],
                }

                for iter_i in range(MAX_ITER):
                    feat = all_features[b, prev].unsqueeze(0)  # (1, D)

                    logits_b = self._network.fc(feat)["logits"][:, :C]

                    p = int(logits_b.argmax(dim=1).item())
                    nxt = int(self.cls2task[p])

                    # per-iter log
                    try:
                        if C > 1:
                            prob_vec = F.softmax(logits_b, dim=1)[0]
                        else:
                            prob_vec = torch.sigmoid(logits_b)[0]
                        conf = float(prob_vec[p].item())

                        topk = min(int(debug_cfg.log_topk), C)
                        topv, topi = prob_vec.topk(topk)
                        trace["iters"].append({
                            "iter": int(iter_i),
                            "prev_t": int(prev),
                            "pred_class": int(p),
                            "pred_task": int(nxt),
                            "conf": conf,
                            "topk": [{"c": int(ci), "p": float(pv)} for ci, pv in zip(topi.tolist(), topv.tolist())],
                        })
                    except Exception:
                        pass

                    if nxt == prev:
                        break
                    prev = nxt

                trace["final_t"] = int(prev)
                trace["n_iters"] = int(len(trace["iters"]))
                routing_traces[b] = trace

                final_logits_list.append(logits_b)

            final_logits = torch.cat(final_logits_list, dim=0).to(self._device)  # (B, C)

            # -------- output probabilities / logits --------
            if return_logits:
                outputs = final_logits
            else:
                if self.ensemble:
                    if is_single_label:
                        final_p = F.softmax(final_logits, dim=1)
                        orig_p = F.softmax(orig_logits, dim=1)
                    else:
                        final_p = torch.sigmoid(final_logits)
                        orig_p = torch.sigmoid(orig_logits)
                    outputs = self.alpha * final_p + (1.0 - self.alpha) * orig_p
                else:
                    outputs = F.softmax(final_logits, dim=1) if is_single_label else torch.sigmoid(final_logits)

            _sync_if_needed(outputs)
            t1 = time.perf_counter()
            batch_infer_time_ms = (t1 - t0) * 1000.0
            infer_time_ms = batch_infer_time_ms / max(1, int(B))

            # ========== END timing ==========
            per_image_time_ms_all.extend([float(infer_time_ms)] * int(B))

            # store predictions/targets
            y_pred_list.append(outputs.detach().cpu().numpy())
            y_true_list.append(targets_seen.detach().cpu().numpy())

            # -------- export prototype graph once (after timing, exclude IO) --------
            if export_graph and (not graph_exported) and run_dir is not None and graph_export_buffer is not None:
                try:
                    sim_b, idx_b, vals_b, A_b, tau_b = graph_export_buffer
                    cls_names = None
                    if hasattr(self, "class_keys"):
                        try:
                            cls_names = [str(k) for k in list(self.class_keys)[:C]]
                        except Exception:
                            cls_names = None

                    out_json = os.path.join(run_dir, "prototype_graph.json") if export_graph_json else None
                    out_csv = os.path.join(run_dir, "prototype_graph.csv") if export_graph_csv else None
                    export_prototype_graph(
                        sim=sim_b,
                        topk_idx=idx_b,
                        topk_vals=vals_b,
                        A=A_b,
                        out_json=out_json,
                        out_csv=out_csv,
                        class_names=cls_names,
                        tau=tau_b,
                    )
                    graph_exported = True
                except Exception:
                    logging.exception("[graph] export failed")

            # -------- per-sample debug logging & visualization (excluded from timing) --------
            if debug_cfg.enabled or viz_enable:
                try:
                    idx_list = indices.detach().cpu().tolist() if torch.is_tensor(indices) else list(indices)
                except Exception:
                    idx_list = list(range(B))

                if is_single_label:
                    if return_logits:
                        prob_final = F.softmax(final_logits, dim=1)
                        pred_cls = prob_final.argmax(dim=1)
                        confs = prob_final.max(dim=1)[0]
                    else:
                        pred_cls = outputs.argmax(dim=1)
                        confs = outputs.max(dim=1)[0]
                else:
                    if return_logits:
                        prob_final = torch.sigmoid(final_logits)
                        pred_bin = (prob_final > 0.5).to(torch.float32)
                        confs = prob_final.mean(dim=1)
                    else:
                        pred_bin = (outputs > 0.5).to(torch.float32)
                        confs = outputs.mean(dim=1)

                for bi in range(B):
                    sample_id = int(idx_list[bi]) if bi < len(idx_list) else int(bi)
                    trace = routing_traces[bi]

                    if is_single_label:
                        true_i = int(targets_seen[bi].item())
                        pred_i = int(pred_cls[bi].item())
                        correct = compute_correct_single_label(pred_i, true_i)
                        conf_i = float(confs[bi].item())
                        multi_stats = None
                    else:
                        true_vec = targets_seen[bi].detach().cpu().numpy().astype(np.float32)
                        pred_vec = pred_bin[bi].detach().cpu().numpy().astype(np.float32)
                        correct = compute_correct_multi_label(pred_vec, true_vec)
                        conf_i = float(confs[bi].item())
                        multi_stats = {
                            "elem_acc": float((pred_vec == true_vec).mean()),
                            "exact_match": bool(correct),
                        }
                        true_i = true_vec.tolist()
                        pred_i = pred_vec.tolist()

                    do_log = debug_cfg.enabled and (debug_cfg.max_samples < 0 or debug_written < debug_cfg.max_samples)
                    if do_log and debug_cfg.only_errors and bool(correct):
                        do_log = False

                    do_viz = viz_enable and (viz_max_samples < 0 or viz_written < viz_max_samples)
                    if do_viz and viz_only_errors and bool(correct):
                        do_viz = False

                    if do_log and debug_writer is not None and trace is not None:
                        rec = {
                            "task_id": int(self._cur_task),
                            "batch_idx": int(batch_idx),
                            "sample_index": sample_id,
                            "true": true_i,
                            "pred": pred_i,
                            "conf": conf_i,
                            "correct": bool(correct),
                            "routing": trace,
                            "ensemble": bool(self.ensemble),
                            "multi_label_stats": multi_stats,
                            "infer_time_ms": float(infer_time_ms),
                            "batch_infer_time_ms": float(batch_infer_time_ms),
                            "batch_size": int(B),
                            "timing_mode": "batch_avg_per_image",
                            "prompt_tag": str(prompt_tag),
                            "prompt_json_path": str(prompt_json_path),
                        }
                        debug_writer.write(rec)
                        debug_written += 1

                    if do_viz and trace is not None and is_single_label and run_dir is not None:
                        try:
                            final_t = int(trace.get("final_t", trace.get("init_best_t", 0)))
                            pil = tensor_to_pil(inputs[bi])

                            viz_dir = os.path.join(run_dir, "viz")
                            os.makedirs(viz_dir, exist_ok=True)

                            base_id = f"idx{sample_id}"
                            tag = f"{base_id}_true{true_i}_pred{pred_i}_t{final_t}"

                            # ---- save RAW (file name only uses idx, stable across different prompts) ----
                            raw_png = os.path.join(viz_dir, base_id + "_raw.png")
                            raw_pdf = os.path.join(viz_dir, base_id + "_raw.pdf")

                            if not os.path.exists(raw_png):
                                if _HAS_SAVE_RAW:
                                    save_raw_image(
                                        pil_img=pil,
                                        out_png=raw_png,
                                        out_pdf=raw_pdf,
                                        title="raw",
                                        dpi=viz_dpi,
                                    )
                                else:
                                    # fallback: PIL direct save
                                    os.makedirs(os.path.dirname(raw_png), exist_ok=True)
                                    pil.save(raw_png)
                                    try:
                                        pil.convert("RGB").save(raw_pdf, "PDF")
                                    except Exception:
                                        pass

                            # ---- rollout ----
                            if viz_use_rollout:
                                heat = attention_rollout(self._network.backbone, inputs[bi:bi + 1], adapter_id=final_t)
                                save_overlay(
                                    pil_img=pil,
                                    heatmap=heat[0],
                                    out_png=os.path.join(viz_dir, tag + "_attn_rollout.png"),
                                    out_pdf=os.path.join(viz_dir, tag + "_attn_rollout.pdf"),
                                    alpha=viz_alpha,
                                    title="attn_rollout",
                                    dpi=viz_dpi,
                                )
                                viz_written += 1

                            # ---- patch-text cosine heatmap ----
                            if viz_use_patch_text and hasattr(self, "class_proto_tensor"):
                                cls_id = int(true_i) if viz_patch_target == "true" else int(pred_i)
                                cls_id = max(0, min(cls_id, C - 1))
                                text_vec = self.class_proto_tensor[cls_id].to(inputs.device)

                                heat = patch_text_cosine_heatmap(
                                    self._network.backbone,
                                    inputs[bi:bi + 1],
                                    adapter_id=final_t,
                                    text_proto=text_vec
                                )
                                save_overlay(
                                    pil_img=pil,
                                    heatmap=heat[0],
                                    out_png=os.path.join(viz_dir, tag + f"_patch_text_{viz_patch_target}.png"),
                                    out_pdf=os.path.join(viz_dir, tag + f"_patch_text_{viz_patch_target}.pdf"),
                                    alpha=viz_alpha,
                                    title="patch_text_cosine",
                                    dpi=viz_dpi,
                                )
                                viz_written += 1

                            # ---- ViT-GradCAM (needs grad even under @no_grad) ----
                            if viz_use_gradcam:
                                with torch.enable_grad():
                                    x1 = inputs[bi:bi + 1].detach().clone().requires_grad_(True)
                                    heat = vit_gradcam_last_attn(
                                        self._network.backbone,
                                        x1,
                                        adapter_id=final_t,
                                        target_class=int(pred_i)
                                    )
                                save_overlay(
                                    pil_img=pil,
                                    heatmap=heat[0],
                                    out_png=os.path.join(viz_dir, tag + "_vit_gradcam.png"),
                                    out_pdf=os.path.join(viz_dir, tag + "_vit_gradcam.pdf"),
                                    alpha=viz_alpha,
                                    title="vit_gradcam",
                                    dpi=viz_dpi,
                                )
                                viz_written += 1

                        except Exception:
                            logging.exception(f"[viz] failed to export sample {sample_id}")

        y_pred = np.concatenate(y_pred_list, axis=0)

        if len(y_true_list) > 0 and np.asarray(y_true_list[0]).ndim == 1:
            y_true = np.concatenate(y_true_list, axis=0)  # (N,)
        else:
            y_true = np.concatenate(y_true_list, axis=0)  # (N, C)

        orig_score = np.concatenate(orig_score_list, axis=0)

        # -------- logging: original model accuracy --------
        if y_true.ndim == 1:
            orig_pred = orig_score.argmax(axis=1)
            orig_acc = (orig_pred == y_true).mean() * 100
            logging.info(f"the top-1 accuracy of the original model: {orig_acc:.2f}")
        else:
            orig_bin = (orig_score > 0.5).astype(np.float32)
            orig_acc = (orig_bin == y_true).mean() * 100
            logging.info(f"the element-wise accuracy of the original model: {orig_acc:.2f}")

        # -------- logging: per-image inference time stats --------
        if len(per_image_time_ms_all) > 0:
            arr = np.array(per_image_time_ms_all, dtype=np.float32)
            logging.info(
                f"[timing] per-image infer_time_ms (batch_avg_per_image): "
                f"mean={arr.mean():.3f}, median={np.median(arr):.3f}, p95={np.percentile(arr, 95):.3f}, n={arr.size}"
            )

        if debug_writer is not None:
            debug_writer.__exit__(None, None, None)

        return (y_pred, y_true)
