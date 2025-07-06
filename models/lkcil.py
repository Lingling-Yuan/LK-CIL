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

        # Hyperparameters following the paper notation
        self.alpha = args.get('alpha', 0.5)         # Base scale α for ensemble blending
        self.beta = args.get('beta', 0.3)           # Fusion coefficient β for ensemble blending
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
        self.ensemble = args["ensemble"]  # Whether to ensemble during inference

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
                label = label.to(self._device)
                # Extract features from original network
                embedding = model.forward_orig(data)['features']
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        # Compute class-wise prototypes and assign to fc weights
        class_list = np.unique(self.train_dataset.labels)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            e = embedding_list[data_index]
            proto = e.mean(0)
            self._network.fc.weight.data[class_index] = proto
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

        # Step 0: Prepare text prototypes for all seen classes
        all_ids = sorted(
            list(self.txt_mgr.prompts.keys()),
            key=lambda k: int(k.split('_')[0])
        )
        assert len(all_ids) >= self._total_classes
        self.class_keys = all_ids[:self._total_classes]
        self.class_proto_tensor = self.txt_mgr.get_class_proto_tensor(self.class_keys)

        # Map new classes to current task index
        for i in range(self._known_classes, self._total_classes):
            self.cls2task[i] = self._cur_task

        # Expand classifier to accommodate new classes
        self._network.update_fc(self._total_classes)
        logging.info(f"Learning on {self._known_classes}-{self._total_classes}")

        # Create data loaders for training and testing
        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        # Separate loader for prototype computation
        train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="test"
        )
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet,
            batch_size=self.batch_size, shuffle=True, num_workers=num_workers
        )

        # Sanity checks: print a few mapped class IDs
        for idx, class_id in enumerate(self.class_keys):
            print(f"[sanity] label {idx} ⇒ class_id '{class_id}'")
        first_batch = next(iter(self.train_loader))
        _, inputs, targets = first_batch
        print(f"[sanity] sample targets: {targets[:10].tolist()}")
        print(f"[sanity] sample class_ids: {[self.class_keys[t] for t in targets[:10].cpu().tolist()]}")

        # If multi-GPU, wrap model in DataParallel
        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        # 1) Adapter fine-tuning on current task
        self._train(self.train_loader, self.test_loader)
        # 2) Replace fc weights with prototypes
        self.replace_fc()
        # 3) Restore single-GPU model if needed
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
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for epoch in prog_bar:
            self._network.backbone.train()
            losses, correct, total = 0.0, 0, 0
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device).long()
                # Forward pass through network with current adapter
                output = self._network(inputs, adapter_id=self._cur_task, train=True)
                logits = output['logits'][:, :self._total_classes]
                # Prevent old-class bias by masking logits
                logits[:, :self._known_classes] = float('-inf')
                # Cross-entropy loss
                ce = F.cross_entropy(logits, targets)
                # Orthogonality loss to diversify features
                orth = self.orth_loss(output['pre_logits'], targets)
                # Alignment loss to enforce text-visual consistency
                v = output['pre_logits']
                text_proto = self.class_proto_tensor[targets]
                cos_sim = F.cosine_similarity(v, text_proto, dim=1)
                align = (1.0 - cos_sim).mean()
                # Total loss
                loss = ce + orth + self.lambda_align * align
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)
            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            prog_bar.set_description(
                f"Task {self._cur_task}, Epoch {epoch+1}/{self.args['tuned_epoch']} => "
                f"Loss {losses/len(train_loader):.3f}, TrainAccy {train_acc:.2f}")
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
                np.arange(class_idx, class_idx + 1), source="train", mode="test", ret_data=True
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size * 3, shuffle=False, num_workers=4)
            vectors = []
            for _, _inputs, _ in idx_loader:
                feats = model(_inputs.to(self._device), adapter_id=self._cur_task, train=True)["features"]
                vectors.append(feats)
            vectors = torch.cat(vectors, dim=0)
            if self.args["ca_storage_efficient_method"] in ['covariance', 'variance']:
                self.cls_mean[class_idx] = vectors.mean(dim=0).to(self._device)
                cov = torch.cov(vectors.T) + torch.eye(vectors.size(1), device=self._device) * 1e-4
                self.cls_cov[class_idx] = cov if self.args["ca_storage_efficient_method"] == 'covariance' else torch.diag(cov)
            else:
                from sklearn.cluster import KMeans
                arr = vectors.cpu().numpy()
                kmeans = KMeans(n_clusters=self.args["n_centroids"], n_init='auto').fit(arr)
                means, vars_ = [], []
                for i in range(self.args["n_centroids"]):
                    cluster = arr[kmeans.labels_ == i]
                    means.append(torch.tensor(cluster.mean(0), device=self._device))
                    vars_.append(torch.tensor(cluster.var(0), device=self._device))
                self.cls_mean[class_idx], self.cls_cov[class_idx] = means, vars_
        # Stack visual means and combine with text prototypes
        vis_proto = torch.stack([self.cls_mean[c] for c in range(self._total_classes)], dim=0)
        text_proto = self.class_proto_tensor
        proto_2d = torch.cat([vis_proto, text_proto], dim=1)
        proto_d = self.proto_proj(proto_2d)
        # Normalize to get final class prototypes
        self.cat_proto = F.normalize(proto_d, p=2, dim=1)

    def classifer_align(self, model):
        """
        Classifier alignment: generate synthetic samples via multivariate normal
        and fine-tune classifier head to correct bias across classes.
        """
        model.train()
        param_list = [p for n, p in model.named_parameters() if p.requires_grad and 'adapter' not in n]
        optimizer = optim.SGD([
            {'params': param_list, 'lr': self.ca_lr, 'weight_decay': self.weight_decay}],
            momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.crct_epochs)
        prog_bar = tqdm(range(self.crct_epochs))
        num_sampled = self.batch_size * 5
        for epoch in prog_bar:
            sampled_data, sampled_label = [], []
            if self.args["ca_storage_efficient_method"] in ['covariance', 'variance']:
                for class_idx in range(self._total_classes):
                    mean, cov = self.cls_mean[class_idx], self.cls_cov[class_idx]
                    m = MultivariateNormal(mean.float(), cov.float())
                    sampled = m.sample((num_sampled,))  # synthetic draws
                    sampled_data.append(sampled)
                    sampled_label.extend([class_idx]*num_sampled)
            else:
                raise NotImplementedError
            inputs = torch.cat(sampled_data).to(self._device)
            targets = torch.tensor(sampled_label).long().to(self._device)
            sf = torch.randperm(inputs.size(0))
            inputs, targets = inputs[sf], targets[sf]
            losses, correct, total = 0.0, 0, 0
            for i in range(self._total_classes):
                inp = inputs[i*num_sampled:(i+1)*num_sampled]
                tgt = targets[i*num_sampled:(i+1)*num_sampled]
                outputs = model(inp, fc_only=True)["logits"][:, :self._total_classes]
                loss = F.cross_entropy(outputs, tgt)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                _, preds = torch.max(outputs, dim=1)
                correct += preds.eq(tgt).cpu().sum(); total += len(tgt)
                losses += loss.item()
            scheduler.step()
            acc = np.round(tensor2numpy(correct)*100/total,2)
            prog_bar.set_description(
                f"Task {self._cur_task}, CA Epoch {epoch+1}/{self.crct_epochs} => "
                f"Loss {losses/self._total_classes:.3f}, CAAcc {acc:.2f}")
        logging.info(prog_bar)

    def orth_loss(self, features, targets):
        """
        Orthogonality loss: encourage feature diversity by pushing apart samples.
        If class means available, incorporate them for extra regularization.
        """
        if self.cls_mean:
            # Stack class means and current features
            sample_mean = torch.stack([
                v for v in self.cls_mean.values()
            ] if not isinstance(list(self.cls_mean.values())[0], list) else
                [m for means in self.cls_mean.values() for m in means], dim=0).to(self._device)
            M = torch.cat([sample_mean, features], dim=0)
            sim = torch.matmul(M, M.t()) / 0.8
            loss = F.cross_entropy(sim, torch.arange(sim.size(0)).long().to(self._device))
            return self.lambda_orth * loss
        else:
            sim = torch.matmul(features, features.t()) / 0.8
            loss = F.cross_entropy(sim, torch.arange(sim.size(0)).long().to(self._device))
            return self.lambda_orth * loss

    @torch.no_grad()
    def _eval_cnn(self, loader, return_logits=False):
        """
        Evaluation with graph-based prototype refinement and optional ensemble.
        """
        # Reset adapter to merged global state
        self._network.backbone.cur_adapter = self._network.backbone.global_adapter
        self._network.eval()
        y_pred, y_true, orig_y_pred = [], [], []
        cos = nn.CosineSimilarity(dim=-1)
        MAX_ITER, C = 4, self._total_classes
        for _, (_, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self._device), targets.flatten().to(self._device)
            orig_logits = self._network.forward_orig(inputs)["logits"][:, :C]
            orig_y_pred.append(orig_logits.argmax(dim=1).cpu().numpy())
            B, T, D = inputs.size(0), self._cur_task+1, self._network.backbone.out_dim
            all_features = torch.zeros(B, T, D, device=self._device)
            # Extract features from each adapter (task) stage
            for t in range(T):
                all_features[:, t, :] = self._network.backbone(
                    inputs, adapter_id=t, train=False)["features"]
            X = self.cat_proto  # Normalized prototypes
            sim = cos(X.unsqueeze(1).expand(-1, C, -1), X.unsqueeze(0).expand(C, -1, -1))
            K = min(self.args.get("graph_sparsity_k", 5), C)
            tau = self.args.get("graph_tau", 1.0)
            mask = torch.full_like(sim, float('-inf'))
            vals, idx = sim.topk(K, dim=1)
            mask.scatter_(1, idx, vals)
            # Softmax-normalized adjacency
            A = F.softmax(mask / tau, dim=1)
            # Two-layer graph refinement
            W1, b1 = self.graph_selector[0].weight, self.graph_selector[0].bias
            H1 = F.relu(A @ X @ W1.t() + b1)
            W2, b2 = self.graph_selector[2].weight, self.graph_selector[2].bias
            H = A @ H1 @ W2.t() + b2
            # Map classes to task indices for refinement iteration
            task_idx = torch.tensor([self.cls2task[c] for c in range(C)], device=self._device)
            flat_feats = all_features.reshape(B * C, D)
            flat_H = H.unsqueeze(0).expand(B, -1, -1).reshape(B * C, D)
            sims_cat = cos(flat_feats, flat_H).view(B, C)
            refined_cat_idx = sims_cat.argmax(dim=1)
            refined_idx = task_idx[refined_cat_idx]
            final_logits = []
            # Self-refinement iteration up to MAX_ITER
            for b in range(B):
                prev = int(refined_idx[b])
                for _ in range(MAX_ITER):
                    feat = all_features[b, prev].unsqueeze(0)
                    logits_b = self._network.backbone(
                        feat, fc_only=True)["logits"][:, :C]
                    p = logits_b.argmax(dim=1).item()
                    nxt = self.cls2task[p]
                    if nxt == prev:
                        break
                    prev = nxt
                final_logits.append(logits_b)
            final_logits = torch.cat(final_logits, dim=0).to(self._device)
            # Optional ensemble with original logits
            if self.ensemble:
                final_p = F.softmax(final_logits, dim=1)
                orig_p = F.softmax(orig_logits, dim=1)
                outputs = self.alpha * final_p + self.beta * orig_p
            else:
                outputs = final_logits
            preds = outputs.topk(min(self.topk, outputs.size(1)), dim=1)[1]
            y_pred.append(preds.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        orig_acc = (np.concatenate(orig_y_pred) == np.concatenate(y_true)).mean() * 100
        logging.info(f"the accuracy of the original model: {orig_acc:.2f}")
        return (np.concatenate(y_pred), np.concatenate(y_true))
