import json
import numpy as np
import torch
import torch.nn.functional as F

class TextFeatureManager:
    def __init__(self, prompts_path, text_feats_path, keys_path, device="cuda"):
        # prompts.json 里的 class_id → [descriptions]
        with open(prompts_path, 'r', encoding='utf-8') as f:
            self.prompts = json.load(f)
        # text_feats.npy: (num_prompts, dim)
        self.text_feats = np.load(text_feats_path)            # np.ndarray
        # text_feats_keys.json: list of class_id strings, same order as rows of text_feats
        self.keys = json.load(open(keys_path, 'r', encoding='utf-8'))
        # 转 torch tensor，归一化
        feats = torch.from_numpy(self.text_feats).float().to(device)
        self.text_feats = F.normalize(feats, dim=1)
        self.device = device

        # 计算每个类的“知识原型” K_c
        self.class_prototypes = self._compute_class_prototypes()
        # 初始化全局任务原型为 None
        self.global_task_proto = None
        self.alpha_base = 0.5  # 默认的 base merge rate，可改

    def _compute_class_prototypes(self):
        """将同一类别的若干文本向量平均并归一化 → {class_id: Tensor(dim)}"""
        proto = {}
        for class_id in self.prompts:
            # 找到 keys 中所有属于本 class_id 的索引
            idxs = [i for i,k in enumerate(self.keys) if k == class_id]
            if not idxs:
                continue
            vecs = self.text_feats[idxs]              # (n_desc, dim)
            mean = vecs.mean(dim=0, keepdim=True)     # (1,dim)
            proto[class_id] = F.normalize(mean, dim=1).squeeze(0)
        return proto

    def get_class_proto(self, class_ids):
        """给 batch 中的 class_ids 返回 [B, dim] 的 proto 矩阵"""
        # class_ids: Tensor([B]) of strings or ints mapping to prompts keys
        # 假设你能把标签 id 转为 class_id string
        protos = [ self.class_prototypes[c] for c in class_ids ]
        return torch.stack(protos, dim=0).to(self.device)

    def update_task_proto(self, new_class_ids):
        """在合并阶段，为任务 i 计算单一任务原型 K_i"""
        protos = [ self.class_prototypes[c] for c in new_class_ids ]
        K_i = torch.stack(protos, dim=0).mean(dim=0)  # (dim,)
        K_i = F.normalize(K_i, dim=0)
        return K_i

    def compute_merge_alpha(self, K_i):
        """给定本次任务原型 K_i，与 self.global_task_proto 计算动态 alpha_i"""
        if self.global_task_proto is None:
            # 第一个任务时，直接用 base
            alpha_i = self.alpha_base
            self.global_task_proto = K_i.clone()
            return alpha_i

        # 计算余弦距离
        cos = torch.dot(K_i, self.global_task_proto)
        d_i = 1.0 - cos.clamp(-1,1)
        # 动态 alpha
        alpha_i = self.alpha_base / (1.0 + d_i)
        # 可选 clamp
        alpha_i = alpha_i.clamp(0.1, 0.9)
        # 更新全局任务原型
        self.global_task_proto = F.normalize((1-alpha_i)*self.global_task_proto + alpha_i*K_i, dim=0)
        print(f"[Merge] compute_merge_alpha -> d_i={d_i:.4f}, alpha_i={alpha_i:.4f}")

        return alpha_i

    def get_class_proto_tensor(self, class_keys):
        """
        输入：按数值标签顺序排列的 class_id 列表（如 ["01_TUMOR","02_STROMA",…]）
        输出：对应 [C, D] 的知识原型张量，可直接用来按 label 索引。
        """
        # 从 class_prototypes dict 中依次取出每个 class_keys 对应的向量
        protos = [self.class_prototypes[c] for c in class_keys]
        # 堆叠成 Tensor 并移动到正确的 device
        return torch.stack(protos, dim=0).to(self.device)
