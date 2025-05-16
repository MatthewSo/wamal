# Weight-Aware Meta Auxiliary Learning (WAMAL)

This repository implements **Weight-Aware Meta Auxiliary Learning (WAMAL)**, a novel framework for improving primary task performance by dynamically learning auxiliary task labels and per-sample auxiliary loss weights using bi-level optimization.

## Features
- Joint learning of auxiliary labels and sample-level auxiliary loss weights.
- Generic wrappers for integrating WAMAL with various image classification backbones.
- Support for VGG16, ResNet50, and Vision Transformer (ViT-B/16) architectures.
- Benchmarked on datasets like CIFAR-10, CIFAR-100, SVHN, Oxford-IIIT Pet, Food101, and CUB200.

## Requirements
- Python 3.8 or higher
- PyTorch
- NumPy
- Additional dependencies listed in `requirements.txt`

Install the required dependencies using:
```bash
pip install -r requirements.txt
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

## File Structure
- `wamal/networks/`: Contains backbone architectures and WAMAL wrappers.
- `wamal/train.py`: Script for training the model.
- `wamal/evaluate.py`: Script for evaluating the model.
- `wamal/utils/`: Utility functions for data preprocessing and training.

## Results
WAMAL demonstrates significant improvements over standard Single-Task Learning (STL) and Meta Auxiliary Learning (MAXL) across various benchmarks. For example:
- **CIFAR-100 (20 Superclasses)**: WAMAL achieves 81.5% test accuracy compared to 76.6% (MAXL) and 75.7% (human-designed auxiliary task).
- **Oxford-IIIT Pet**: WAMAL improves fine-tuning accuracy by 0.46% over MAXL.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.