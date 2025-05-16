from __future__ import annotations
from transformers.modeling_outputs import ModelOutput

import numpy as np
import torch, importlib
from pathlib import Path
from collections import OrderedDict
from typing import Callable, Tuple
import torch.nn as nn
from torch.func import functional_call

def strip_classifier(model: nn.Module, input_shape):
    # torchvision ResNet / RegNet / EfficientNet (fc)
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        dim = model.fc.in_features
        model.fc = nn.Identity()
        return dim
    print(model)

    # VGG / MobileNet / DenseNet / ViT (classifier)
    if hasattr(model, 'classifier'):
        cls = model.classifier
        # Linear
        if isinstance(cls, nn.Linear):
            dim = cls.in_features
            model.classifier = nn.Identity()
            return dim
        # Sequential – last layer Linear
        if isinstance(cls, nn.Sequential):
            last = list(cls.children())[-1]
            if isinstance(last, nn.Linear):
                dim = last.in_features
                model.classifier = nn.Identity()
                return dim
            # Check second last layer
            if len(cls) > 1 and isinstance(cls[-2], nn.Linear):
                dim = cls[-2].in_features
                model.classifier = nn.Identity()
                return dim
    head_names = ["fc", "classifier", "head", "heads"]
    for name in head_names:
        if hasattr(model, name):
            layer = getattr(model, name)

            # torchvision keeps a ModuleDict 'heads', we want its 'head'
            if name == "heads" and isinstance(layer, nn.ModuleDict) and "head" in layer:
                layer = layer["head"]
                setattr(model, "heads", nn.Identity())

            # plain Linear
            if isinstance(layer, nn.Linear):
                dim = layer.in_features
                setattr(model, name, nn.Identity())
                return dim

            # Sequential whose last item is Linear
            if isinstance(layer, nn.Sequential):
                if isinstance(layer[-1], nn.Linear):
                    dim = layer[-1].in_features
                    setattr(model, name, nn.Identity())
                    return dim
    # Fallback: run dummy input to measure
    raise Exception("Warning: Unable to strip classifier from model. Using dummy input to measure feature dimension.")
    dummy = torch.zeros(1, *input_shape)
    with torch.no_grad():
        feat = model(dummy)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        feat = feat.flatten(1)
    return feat.shape[1]


class WamalWrapper(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 num_primary: int,
                 num_auxiliary: int,
                 input_shape: Tuple[int, int, int] = (3, 224, 224)):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = strip_classifier(self.backbone, input_shape)
        self.num_primary = num_primary
        self.num_auxiliary = num_auxiliary
        self.input_shape = input_shape

        self.primary_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, num_primary),
            nn.Softmax(dim=1),
        )

        self.auxiliary_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, num_auxiliary),
            nn.Softmax(dim=1),
        )

    def forward(self,
                x: torch.Tensor,
                params:  OrderedDict | None = None,
                buffers: OrderedDict | None = None,
                **kwargs):
        if params is None and buffers is None:
            feat = self.backbone(x, **kwargs)
            if isinstance(feat, ModelOutput):
                feat = feat.logits
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            feat = feat.flatten(1)
            return (
                self.primary_head(feat),
                self.auxiliary_head(feat),
            )

        params  = params or OrderedDict()
        buffers = buffers or OrderedDict()
        merged  = {**params, **buffers}

        # Backbone overrides
        bb_ov = {k.split('backbone.', 1)[1]: v for k, v in merged.items()
                 if k.startswith('backbone.')}
        feat = functional_call(self.backbone, bb_ov, (x,), kwargs)
        if isinstance(feat, ModelOutput):
            feat = feat.logits
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        feat = feat.flatten(1)

        # Primary head overrides
        pri_ov = {k.split('primary_head.', 1)[1]: v for k, v in merged.items()
                  if k.startswith('primary_head.')}
        primary_logits = functional_call(self.primary_head, pri_ov, (feat,))

        # Auxiliary head overrides
        aux_ov = {k.split('auxiliary_head.', 1)[1]: v for k, v in merged.items()
                  if k.startswith('auxiliary_head.')}
        auxiliary_logits = functional_call(self.auxiliary_head, aux_ov, (feat,))

        return primary_logits, auxiliary_logits

    def save(self, path: str | Path, **extra):
        ckpt = {
            "state_dict"    : self.state_dict(),
            "num_primary"   : self.num_primary,
            "num_auxiliary" : self.num_auxiliary,
            "input_shape"   : self.input_shape,
            "backbone_module": self.backbone.__class__.__module__,
            "backbone_name"  : self.backbone.__class__.__name__,
            "extra"          : extra,
        }
        torch.save(ckpt, Path(path))

    @classmethod
    def load(cls,
             path: str | Path,
             backbone_fn: Callable[[], nn.Module] | None = None,
             map_location: str | torch.device | None = None) -> "WamalWrapper":
        ckpt = torch.load(path, map_location=map_location)

        if backbone_fn is not None:
            backbone = backbone_fn()
        else:
            mod  = importlib.import_module(ckpt["backbone_module"])
            cls_ = getattr(mod, ckpt["backbone_name"])
            backbone = cls_()

        model = cls(backbone=backbone,
                    num_primary=ckpt["num_primary"],
                    num_auxiliary=ckpt["num_auxiliary"],
                    input_shape=tuple(ckpt["input_shape"]))
        model.load_state_dict(ckpt["state_dict"])
        return model

class LabelWeightWrapper(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 num_primary: int,
                 num_auxiliary: int,
                 input_shape: Tuple[int, int, int] = (3, 224, 224)):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = strip_classifier(self.backbone, input_shape)
        self.num_primary = num_primary
        self.num_auxiliary = num_auxiliary
        self.input_shape = input_shape
        self.psi = np.array([num_auxiliary // num_primary] * num_primary)

        self.classifier_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.num_auxiliary),
        )
        self.weight_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _mask_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = 1):
        exp = torch.exp(logits) * mask
        return exp / (exp.sum(dim=dim, keepdim=True) + 1e-12)

    def _build_mask(self, y: torch.Tensor):
        index = torch.zeros(self.num_primary, self.num_auxiliary, device=y.device)
        start = 0
        for i, k in enumerate(self.psi):
            index[i, start:start+k] = 1.0
            start += k
        return index[y]

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                params:  OrderedDict | None = None,
                buffers: OrderedDict | None = None,
                **kwargs):
        if params is None and buffers is None:
            feat = self.backbone(x, **kwargs)
            if isinstance(feat, ModelOutput):
                feat = feat.logits
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            feat = feat.flatten(1)
            logits  = self.classifier_head(feat)
            mask    = self._build_mask(y)
            labels  = self._mask_softmax(logits, mask)
            weights = self.weight_head(feat)
            return labels, weights

        params  = params or OrderedDict()
        buffers = buffers or OrderedDict()
        merged  = {**params, **buffers}

        bb_ov = {k.split('backbone.',1)[1]:v for k,v in merged.items() if k.startswith('backbone.')}
        feat = functional_call(self.backbone, bb_ov, (x,), kwargs)
        if isinstance(feat, ModelOutput):
            feat = feat.logits
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        feat = feat.flatten(1)

        cls_ov = {k.split('classifier_head.',1)[1]:v for k,v in merged.items() if k.startswith('classifier_head.')}
        logits = functional_call(self.classifier_head, cls_ov, (feat,))

        w_ov = {k.split('weight_head.',1)[1]:v for k,v in merged.items() if k.startswith('weight_head.')}
        weights = functional_call(self.weight_head, w_ov, (feat,))

        mask   = self._build_mask(y)
        labels = self._mask_softmax(logits, mask)
        return labels, weights

    def save(self, path: str | Path, **extra):
        ckpt = {
            "state_dict"    : self.state_dict(),
            "num_primary"   : self.num_primary,
            "num_auxiliary" : self.num_auxiliary,
            "input_shape"   : self.input_shape,
            "backbone_module": self.backbone.__class__.__module__,
            "backbone_name"  : self.backbone.__class__.__name__,
            "extra"          : extra,
        }
        torch.save(ckpt, Path(path))

    @classmethod
    def load(cls,
             path: str | Path,
             backbone_fn: Callable[[], nn.Module] | None = None,
             initialization_args: dict | None = None,
             map_location: str | torch.device | None = None) -> "LabelWeightWrapper":
        ckpt = torch.load(path, map_location=map_location)

        if backbone_fn is not None:
            backbone = backbone_fn()
        else:
            mod  = importlib.import_module(ckpt["backbone_module"])
            cls_ = getattr(mod, ckpt["backbone_name"])
            backbone = cls_(**(initialization_args or {}))
        model = cls(backbone=backbone,
                    num_primary=ckpt["num_primary"],
                    num_auxiliary=ckpt["num_auxiliary"],
                    input_shape=tuple(ckpt["input_shape"]))
        model.load_state_dict(ckpt["state_dict"])
        return model