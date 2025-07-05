import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import LKCILNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, target2onehot
from torch.distributions.multivariate_normal import MultivariateNormal
from utils.text_feature_manager import TextFeatureManager
import copy

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self._network = LKCILNet(args, True)
        self._network.backbone.global_adapter = copy.deepcopy(self._network.backbone.cur_adapter)

        self.txt_mgr = TextFeatureManager(
            prompts_path=args['prompt_json_path'],
            text_feats_path=args['text_feats_path'],
            keys_path=args['text_feats_keys_path'],
            device=self._device
        )

        # hyper-parameters per paper notation:
        self.alpha = args.get('alpha', 0.5)  # base scale α
        self.beta = args.get('beta', 0.3)  # fusion coefficient β
        self.lambda_orth = args.get('lambda_orth', 0.3)  # orthogonal weight λₒᵣₜₕ
        self.lambda_align = args.get('lambda_align', 0.3)  # alignment weight λₐₗᵢgₙ

        D = self._network.backbone.out_dim
        hidden_dim = args.get("graph_hidden_dim", 128)
        self.proto_proj = nn.Linear(2 * D, D).to(self._device)
        self.graph_selector = nn.Sequential(
            nn.Linear(D, hidden_dim),  # D → hidden
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, D),  # hidden → D
        ).to(self._device)

        self.cls_mean = {}
        self.cls_cov = {}
        self.cls2task = {}

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.ca_lr = args["ca_lr"]
        self.crct_epochs = args["crct_epochs"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.ensemble = args["ensemble"]

        for n, p in self._network.backbone.named_parameters():
            if 'adapter' not in n and 'head' not in n:
                p.requires_grad = False

        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable:,} model training parameters.')

    def replace_fc(self):
        model = self._network.to(self._device)
        embedding_list, label_list = [], []
        with torch.no_grad():
            for _, data, label in self.train_loader_for_protonet:
                data, label = data.to(self._device), label.to(self._device)
                emb = model.forward_orig(data)['features']
                embedding_list.append(emb.cpu())
                label_list.append(label.cpu())
        E = torch.cat(embedding_list, dim=0)
        L = torch.cat(label_list, dim=0)
        for cls in np.unique(self.train_dataset.labels):
            idx = (L == cls).nonzero().squeeze(-1)
            proto = E[idx].mean(0)
            self._network.fc.weight.data[cls] = proto
        return model

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._network = self._network.to(self._device)
        self._network.backbone.cur_adapter = copy.deepcopy(self._network.backbone.global_adapter)
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        all_ids = sorted(self.txt_mgr.prompts.keys(), key=lambda k: int(k.split('_')[0]))
        assert len(all_ids) >= self._total_classes, f"Need {self._total_classes} keys but got {len(all_ids)}"
        self.class_keys = all_ids[:self._total_classes]
        self.class_proto_tensor = self.txt_mgr.get_class_proto_tensor(self.class_keys)

        for i in range(self._known_classes, self._total_classes):
            self.cls2task[i] = self._cur_task

        self._network.update_fc(self._total_classes)
        logging.info(f"Learning on {self._known_classes}-{self._total_classes}")

        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="train"
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=num_workers
        )
        pro_ds = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="test"
        )
        self.train_loader_for_protonet = DataLoader(
            pro_ds, batch_size=self.batch_size,
            shuffle=True, num_workers=num_workers
        )

        self._train(self.train_loader, self.test_loader)
        self.replace_fc()

    def _train(self, train_loader, test_loader):
        self._network.backbone.to(self._device)
        optimizer = self.get_optimizer(self._network.backbone)
        scheduler = self.get_scheduler(optimizer)
        prog = tqdm(range(self.args['tuned_epoch']))
        for epoch in prog:
            self._network.backbone.train()
            total_loss, correct, total = 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device).long()
                out = self._network(inputs, adapter_id=self._cur_task, train=True)
                logits = out["logits"][:, :self._total_classes]
                logits[:, :self._known_classes] = float('-inf')

                ce = F.cross_entropy(logits, targets)
                orth = self.orth_loss(out['pre_logits'], targets)
                v = out['pre_logits']
                text_proto = self.class_proto_tensor[targets]
                align = (1 - F.cosine_similarity(v, text_proto, dim=1)).mean()

                # total loss (Eq.12): L_CE + λₒᵣₜₕ·L_orth + λₐₗᵢgₙ·L_align
                # Language-guided Knowledge Merge = align term
                loss = ce + self.lambda_orth * orth + self.lambda_align * align

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)

            if scheduler:
                scheduler.step()
            acc = correct * 100.0 / total
            prog.set_description(
                f"Task {self._cur_task} Ep{epoch + 1} Loss{total_loss / len(train_loader):.3f} Acc{acc:.2f}")
        logging.info(f"Finished training Task {self._cur_task}")

    def get_optimizer(self, model):
        ap = [p for n, p in model.named_parameters() if 'adapter' in n and p.requires_grad]
        hp = [p for n, p in model.named_parameters() if 'adapter' not in n and p.requires_grad]
        params = [
            {'params': ap, 'lr': self.init_lr, 'weight_decay': self.weight_decay},
            {'params': hp, 'lr': self.init_lr * 0.1, 'weight_decay': self.weight_decay}
        ]
        opt = optim.SGD(params, momentum=0.9) if self.args['optimizer'] == 'sgd' else optim.AdamW(params)
        return opt

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        return None

    def classifer_align(self, model):
        model.train()

        run_epochs = self.crct_epochs
        param_list = [p for n, p in model.named_parameters() if p.requires_grad and 'adapter' not in n]
        network_params = [{'params': param_list, 'lr': self.ca_lr, 'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=self.ca_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        prog_bar = tqdm(range(run_epochs))
        for epoch in prog_bar:

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.batch_size * 5

            if self.args["ca_storage_efficient_method"] in ['covariance', 'variance']:
                for class_idx in range(self._total_classes):
                    mean = self.cls_mean[class_idx].to(self._device)
                    cov = self.cls_cov[class_idx].to(self._device)
                    if self.args["ca_storage_efficient_method"] == 'variance':
                        cov = torch.diag(cov)
                    m = MultivariateNormal(mean.float(), cov.float())
                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                    sampled_data.append(sampled_data_single)

                    sampled_label.extend([class_idx] * num_sampled_pcls)

            elif self.args["ca_storage_efficient_method"] == 'multi-centroid':
                for class_idx in range(self._total_classes):
                    for cluster in range(len(self.cls_mean[class_idx])):
                        mean = self.cls_mean[class_idx][cluster]
                        var = self.cls_cov[class_idx][cluster]
                        if var.mean() == 0:
                            continue
                        m = MultivariateNormal(mean.float(), (
                                    torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)).float())
                        sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                        sampled_data.append(sampled_data_single)
                        sampled_label.extend([class_idx] * num_sampled_pcls)
            else:
                raise NotImplementedError

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)
            if epoch == 0:
                print(f"[CA][Task {self._cur_task}] sampled_data shape: {sampled_data.shape}, "
                      f"labels unique: {torch.unique(sampled_label).tolist()}")

            inputs = sampled_data
            targets = sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            losses = 0.0
            correct, total = 0, 0
            for _iter in range(self._total_classes):
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                outputs = model(inp, fc_only=True)
                logits = outputs['logits'][:, :self._total_classes]

                loss = F.cross_entropy(logits, tgt)

                _, preds = torch.max(logits, dim=1)

                correct += preds.eq(tgt.expand_as(preds)).cpu().sum()
                total += len(tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss

            scheduler.step()
            ca_acc = np.round(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, CA_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.crct_epochs,
                losses / self._total_classes,
                ca_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def orth_loss(self, features, targets):
        if self.cls_mean:
            # orth loss of this batch
            sample_mean = []
            for v in self.cls_mean.values():
                if isinstance(v, list):
                    sample_mean.extend(v)
                else:
                    sample_mean.append(v)
            sample_mean = torch.stack(sample_mean, dim=0).to(self._device, non_blocking=True)
            M = torch.cat([sample_mean, features], dim=0)
            sim = torch.matmul(M, M.t()) / 0.8
            labels = torch.arange(sim.size(0), device=self._device).long()
            loss = F.cross_entropy(sim, labels)
        else:
            sim = torch.matmul(features, features.t()) / 0.8
            labels = torch.arange(sim.size(0), device=self._device).long()
            loss = F.cross_entropy(sim, labels)

        # use lambda_orth instead of args["reg"]
        return self.lambda_orth * loss

    @torch.no_grad()
    def _eval_cnn(self, loader, return_logits=False):
        self._network.backbone.cur_adapter = self._network.backbone.global_adapter
        self._network.eval()

        y_pred, y_true, orig_y = [], [], []
        cos = nn.CosineSimilarity(dim=-1)
        C = self._total_classes

        for _, inputs, targets in loader:
            inputs, targets = inputs.to(self._device), targets.flatten().to(self._device)

            orig_logits = self._network.forward_orig(inputs)["logits"][:, :C]
            orig_preds = orig_logits.argmax(dim=1)
            orig_y.append(orig_preds.cpu().numpy())

            B, T, D = inputs.size(0), self._cur_task + 1, self._network.backbone.out_dim
            all_f = torch.zeros(B, T, D, device=self._device)
            for t in range(T):
                feats = self._network.backbone(inputs, adapter_id=t, train=False)["features"]
                all_f[:, t, :] = feats

            # Graph-based Prototype Refinement (Eq.8)
            X = self.cat_proto  # [C, D]
            sim = cos(X.unsqueeze(1).expand(-1, C, -1), X.unsqueeze(0).expand(C, -1, -1))
            K = min(self.args.get("graph_sparsity_k", 5), C)
            tau = self.args.get("graph_tau", 1.0)
            mask = torch.full_like(sim, float('-inf'))
            vals, idx = sim.topk(K, dim=1)
            mask.scatter_(1, idx, vals)
            A = F.softmax(mask / tau, dim=1)
            W1, b1 = self.graph_selector[0].weight, self.graph_selector[0].bias
            H1 = F.relu(A @ X @ W1.t() + b1)
            W2, b2 = self.graph_selector[2].weight, self.graph_selector[2].bias
            refined_proto = A @ H1 @ W2.t() + b2

            if self.ensemble:
                alpha = self.alpha  # base fusion α
                beta = self.beta  # fusion complement β
                orig_p = F.softmax(orig_logits, dim=1)
                ref_p = F.softmax(refined_proto.unsqueeze(0).expand(B, -1, -1)[..., :C], dim=1)
                outputs = alpha * ref_p + beta * orig_p
            else:
                outputs = orig_logits

            preds = outputs.argmax(dim=1)
            y_pred.append(preds.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)
