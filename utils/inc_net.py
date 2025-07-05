import torch
from torch import nn
import timm
from backbone.linears import CosineLinear
from easydict import EasyDict

def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    if name in ("pretrained_vit_b16_224", "vit_base_patch16_224"):
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif name in ("pretrained_vit_b16_224_in21k", "vit_base_patch16_224_in21k"):
        model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif "_lkcil" in name and args.get("model_name") == "lkcil":
        from backbone import vit_lkcil

        tuning_config = EasyDict(
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=args["ffn_num"],
            d_model=768,
            _device=args["device"][0],
            adapter_momentum=args["adapter_momentum"],
            vpt_on=False,
            vpt_num=0,
        )
        if name == "vit_base_patch16_224_lkcil":
            model = vit_lkcil.vit_base_patch16_224_lkcil(
                num_classes=args["nb_classes"],
                global_pool=False,
                drop_path_rate=0.0,
                tuning_config=tuning_config,
            )
        elif name == "vit_base_patch16_224_in21k_lkcil":
            model = vit_lkcil.vit_base_patch16_224_in21k_lkcil(
                num_classes=args["nb_classes"],
                global_pool=False,
                drop_path_rate=0.0,
                tuning_config=tuning_config,
            )
        else:
            raise NotImplementedError(f"Unknown LK-CIL backbone variant: {name}")
        return model
    else:
        raise NotImplementedError(f"Unknown backbone type: {name}")

class LKCILNet(nn.Module):
    """
    LK-CIL network based on cosine classifier and ViT backbone.
    """

    def __init__(self, args, pretrained=False):
        super().__init__()
        self.backbone = get_backbone(args, pretrained)
        self.backbone.out_dim = 768
        self.fc = None
        self._device = args["device"][0]

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = CosineLinear(self.feature_dim, nb_classes)
        if self.fc is not None:
            prev_out = self.fc.out_features
            old_weight = self.fc.weight.data
            old_sigma = getattr(self.fc, "sigma", None)
            if old_sigma is not None:
                fc.sigma.data = old_sigma.data
            if nextperiod_initialization is not None:
                new_weight = torch.cat([old_weight, nextperiod_initialization], dim=0)
            else:
                zeros = torch.zeros(nb_classes - prev_out, self.feature_dim, device=old_weight.device)
                new_weight = torch.cat([old_weight, zeros], dim=0)
            fc.weight = nn.Parameter(new_weight)
        self.fc = fc.to(self._device).requires_grad_(False)

    def forward_orig(self, x):
        features = self.backbone(x, adapter_id=0)["features"]
        logits_dict = self.fc(features)
        return {
            "features": features,
            "logits": logits_dict["logits"] if isinstance(logits_dict, dict) else logits_dict,
        }

    def forward(self, x, adapter_id=-1, train=False, fc_only=False):
        return self.backbone(x, adapter_id, train, fc_only)
