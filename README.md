# Weight-Aware Meta Auxiliary Learning (WAMAL)

This repository implements **Weight-Aware Meta Auxiliary Learning (WAMAL)**, a novel framework for improving primary task performance by dynamically learning auxiliary task labels and per-sample auxiliary loss weights using bi-level optimization.

## Features
- Joint learning of auxiliary labels and sample-level auxiliary loss weights.
- Generic wrappers for integrating WAMAL with various image classification backbones.
- Support for VGG16, ResNet50, and Vision Transformer (ViT-B/16) architectures.
- Benchmarked on datasets like CIFAR-10, CIFAR-100, SVHN, Oxford-IIIT Pet, Food101, and CUB200.

## Requirements
- Python 3.8 or higher
- Additional dependencies listed in `environment.yaml`

Install the required dependencies using:
```bash
conda env create -f environment.yaml
```

## Usage
To run the entry points in the `wamal` module, use the following command format:
```bash
python -m wamal.<entry_point>.py
```

Replace `<entry_point>` with the specific script you want to execute.

### Example
```bash
python -m wamal.vgg16_cifar_10_maxl.py
```

## Datasets
Datasets required for training and evaluation are **not included** in this repository. You may need to download the datasets manually and ensure they are in the correct format and directory structure as expected by the scripts.

### Supported Datasets
- **CIFAR-10** and **CIFAR-100**: Standard image classification datasets.
- **SVHN**: Street View House Numbers dataset.
- **Oxford-IIIT Pet**: Fine-grained classification of pet breeds.
- **Food101**: Large-scale fine-grained food classification.
- **CUB200**: Fine-grained bird species classification.

#### Supported Wrappers

| Wrapper              | Purpose                                                                  | Typical Forward Call                        |
| -------------------- | ------------------------------------------------------------------------ | ------------------------------------------- |
| `LabelWeightWrapper` | Generates **auxiliary labels** *and* a **weight w∈ℝ⁺** for every sample. | `aux_labels, aux_weights = label_net(x, y)` |
| `WamalWrapper`       | Produces **primary** & **auxiliary** logits that are trained jointly.    | `prim_logits, aux_logits = wamal_net(x)`    |

Both wrappers accept any **torchvision‑style backbone** plus:

```python
LabelWeightWrapper(backbone,
                    num_primary=<#classes for main task>,
                    num_auxiliary=<#aux classes>,
                    input_shape=(C,H,W))
```

---

## Sample Training

```python
import torch, torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
from wamal.wrappers import LabelWeightWrapper, WamalWrapper

# -- helpers -----------------------------------------------------------
def inner_sgd_update(model, loss, lr):
    """Return θ₁⁺ by taking a differentiable SGD step on <loss>."""
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    return OrderedDict((n, p - lr * g) for (n, p), g in zip(model.named_parameters(), grads))

def entropy(logits):
    q = F.softmax(logits, dim=1)
    return (-q * q.log()).sum(1).mean()

# -- data --------------------------------------------------------------
train_loader = DataLoader(
    datasets.CIFAR100("data", train=True, download=True,
                      transform=transforms.ToTensor()),
    batch_size=128, shuffle=True, num_workers=4)

device   = "cuda" if torch.cuda.is_available() else "cpu"
backbone = models.resnet18(weights="IMAGENET1K_V1")

# -- networks ----------------------------------------------------------
wamal_net = WamalWrapper(backbone, num_primary=20, num_auxiliary=100,
                         input_shape=(3, 224, 224)).to(device)
label_net = LabelWeightWrapper(backbone, num_primary=20, num_auxiliary=100,
                               input_shape=(3, 224, 224)).to(device)

main_opt = torch.optim.SGD(wamal_net.parameters(), lr=1e-2, momentum=0.9)
gen_opt  = torch.optim.SGD(label_net.parameters(), lr=1e-2, weight_decay=1e-4)
inner_lr = 1e-2  # step‑size for θ₁⁺

# -- bi‑level optimisation --------------------------------------------
for epoch in range(50):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        # ---- Level 1: update θ₁ (backbone) ----
        prim_logits, aux_logits = wamal_net(x)
        aux_labels, aux_weights = label_net(x, y)

        l1_prim = F.cross_entropy(prim_logits, y)
        l1_aux  = F.cross_entropy(aux_logits, aux_labels)
        level1_loss = l1_prim + (aux_weights * l1_aux).mean()

        main_opt.zero_grad()
        # retain_graph=True so gradients flow through to θ₁⁺
        level1_loss.backward(retain_graph=True)
        main_opt.step()

        # ---- Build θ₁⁺ for meta‑update ----
        fast_weights = inner_sgd_update(wamal_net, level1_loss, inner_lr)

        # ---- Level 2: update θ₂ (label generator) ----
        prim_fast, _ = wamal_net.forward(x, params=fast_weights)
        meta_loss = F.cross_entropy(prim_fast, y) + 0.2 * entropy(aux_logits)

        gen_opt.zero_grad()
        meta_loss.backward()
        gen_opt.step()

    print(f"Epoch {epoch+1}: L1={level1_loss.item():.3f} | Meta={meta_loss.item():.3f}")
```

## Results
WAMAL demonstrates significant improvements over standard Single-Task Learning (STL) and Meta Auxiliary Learning (MAXL) across various benchmarks. For example:
- **CIFAR-100 (20 Superclasses)**: WAMAL achieves 81.5% test accuracy compared to 76.6% (MAXL) and 75.7% (human-designed auxiliary task).
- **Oxford-IIIT Pet**: WAMAL improves fine-tuning accuracy by 0.46% over MAXL.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.