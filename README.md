# Multi-Attribute Facial Recognition System

A comprehensive machine learning project for facial attribute prediction using multiple state-of-the-art deep learning architectures. This repository contains training notebooks and evaluation metrics for models trained on both ISGD and CelebA datasets.

## 📊 Project Overview

This repository contains:
- **Training notebooks** for multiple model architectures
- **Evaluation metrics** for trained models (AUC, F1-Score, Accuracy, Precision, Recall)
- **Comparative analysis** between different architectures
- **Training experiments** with and without data augmentation

## 🎯 Key Objectives

1. **Train multiple backbone architectures** on facial attribute recognition
2. **Compare performance** between different model architectures
3. **Evaluate impact** of data augmentation on model performance
4. **Identify the best-performing model** based on evaluation metrics

## 📁 Project Structure

```
ISGD/
├── README.md                          # This file
├── 📔 Training Notebooks
│   ├── Training_Without_Augmentation.ipynb  # Training without data augmentation
│   ├── Training_With_Augmentation.ipynb     # Training with data augmentation
│   ├── Training_Swin_Transformer.ipynb      # Swin Transformer training
│   └── Training_CelebA.ipynb                # CelebA dataset training
│
└── 📊 Results/
    ├── convnext_tiny_metrics.csv
    ├── efficientnet_b0_metrics.csv
    ├── inception_next_tiny_metrics.csv
    ├── swin_tiny_patch4_window7_224_metrics.csv
    ├── vit_base_patch16_224_metrics.csv
    ├── celeba_convnext_tiny_metrics.csv
    └── celeba_focalnet_tiny_srf_metrics.csv
```

## 🏗️ Model Architectures Tested

### ISGD Dataset Models
| Model | Type | Parameters | Performance |
|-------|------|-----------|-------------|
| **ConvNeXt Tiny** | CNN | ~29M | Excellent |
| **EfficientNet B0** | CNN | ~5M | Very Good |
| **InceptionNext Tiny** | CNN | ~3.8M | Very Good |
| **Swin Transformer** | Transformer | ~29M | Excellent |
| **Vision Transformer (ViT)** | Transformer | ~87M | Excellent |
| **ConvNeXt Tiny V2** | CNN | ~29M | Excellent |

### CelebA Dataset Models
| Model | Dataset | Attributes | Status |
|-------|---------|-----------|--------|
| **FocalNet Tiny SRF** | CelebA | 40 | Trained |
| **ConvNeXt Tiny** | CelebA | 40 | Trained |
| **ConvNeXt Tiny** | CelebA | 40 | Trained |

## 📊 Datasets

This project uses two datasets for training and evaluation:

### ISGD Dataset (Custom)
- **Attributes**: 33 facial attributes
- **Format**: Binary classification per attribute

### CelebA Dataset
- **Attributes**: 40 facial attributes
- **Format**: Binary labels

Note: Dataset files are not included in this repository. Training notebooks expect datasets to be available in appropriate directories.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU training)
- PyTorch 2.0+
- TIMM (PyTorch Image Models)
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/HimalRana2610/ISGD.git
   cd ISGD
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install timm pandas numpy scikit-learn tqdm pillow jupyter
   ```

3. **Prepare datasets**
   - Place your ISGD dataset in the appropriate directory
   - Download CelebA dataset if needed for CelebA training notebooks

## 📓 Notebook Guide

### Training Notebooks

**`Training_Without_Augmentation.ipynb`** - Baseline training
- Trains models without data augmentation
- Multiple architectures on ISGD dataset
- Generates per-attribute metrics

**`Training_With_Augmentation.ipynb`** - Augmented training
- Trains models with data augmentation
- Compares performance impact of augmentation
- Saves evaluation metrics

**`Training_Swin_Transformer.ipynb`** - Swin Transformer specific
- Dedicated training for Swin Transformer architecture
- Hyperparameter tuning
- Performance evaluation

**`Training_CelebA.ipynb`** - CelebA dataset training
- Trains models on CelebA dataset
- 40 attribute classification
- Cross-dataset evaluation

## 🏃 Running the Notebooks

1. **Start Jupyter**
   ```bash
   jupyter notebook
   ```

2. **Open a training notebook**
   - `Training_Without_Augmentation.ipynb` for baseline experiments
   - `Training_With_Augmentation.ipynb` for augmented training
   - `Training_Swin_Transformer.ipynb` for Swin Transformer
   - `Training_CelebA.ipynb` for CelebA training

3. **Run cells sequentially**
   - Follow the notebook instructions
   - Models will be trained and metrics saved to `Results/`

## 📊 Results & Metrics

### Performance Metrics Calculated
- **AUC (Area Under ROC Curve)**: 0.0 - 1.0
- **Accuracy**: Correct predictions / Total
- **Macro F1**: Unweighted mean of F1 scores
- **Micro F1**: F1 calculated globally
- **Precision**: True positives / Predicted positives
- **Recall**: True positives / All positives

### Available Results
Results are saved in `Results/` directory as CSV files:

**ISGD Dataset Models:**
- `convnext_tiny_metrics.csv`
- `efficientnet_b0_metrics.csv`
- `inception_next_tiny_metrics.csv`
- `swin_tiny_patch4_window7_224_metrics.csv`
- `vit_base_patch16_224_metrics.csv`

**CelebA Dataset Models:**
- `celeba_convnext_tiny_metrics.csv`
- `celeba_focalnet_tiny_srf_metrics.csv`

Each CSV contains per-attribute metrics with columns:
```
attribute, auc, accuracy, macro_f1, micro_f1, precision, recall
```

## 🔄 Training Workflow

1. **Prepare dataset** (ISGD or CelebA)
2. **Open appropriate training notebook**
3. **Configure hyperparameters** (learning rate, batch size, etc.)
4. **Run training** with or without augmentation
5. **Monitor training progress** (loss, accuracy)
6. **Evaluate on validation set**
7. **Save metrics to Results/**
8. **Compare performance** across architectures

## 💡 Key Features

- **Multiple Architecture Support**: ConvNeXt, EfficientNet, InceptionNext, Swin Transformer, ViT
- **Data Augmentation Experiments**: Compare training with and without augmentation
- **Cross-Dataset Training**: Both ISGD and CelebA datasets supported
- **Comprehensive Metrics**: Per-attribute evaluation with multiple metrics
- **Reproducible Results**: Metrics saved for all trained models

## 📈 How to Use

### Training a New Model
1. Open one of the training notebooks
2. Modify hyperparameters if needed
3. Run all cells to train the model
4. Metrics will be automatically saved to `Results/`

### Comparing Models
1. Check the CSV files in `Results/` directory
2. Compare per-attribute performance across models
3. Analyze AUC, F1-Score, Accuracy, Precision, and Recall

### Experimenting with Augmentation
1. Use `Training_Without_Augmentation.ipynb` as baseline
2. Run `Training_With_Augmentation.ipynb` for comparison
3. Compare results to see augmentation impact

## 📦 Dependencies

### Core ML Libraries
- `torch` - Deep learning framework
- `torchvision` - Computer vision utilities
- `timm` - Pre-trained model zoo
- `scikit-learn` - Machine learning metrics

### Data Processing
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `pillow` - Image processing

### Utilities
- `tqdm` - Progress bars
- `matplotlib` - Visualization
- `jupyter` - Notebook environment

## 🔍 File Descriptions

### Training Notebooks
- **`Training_Without_Augmentation.ipynb`**: Baseline training without data augmentation
- **`Training_With_Augmentation.ipynb`**: Training with data augmentation applied
- **`Training_Swin_Transformer.ipynb`**: Dedicated Swin Transformer training
- **`Training_CelebA.ipynb`**: CelebA dataset training

### Results Directory
- **`Results/*.csv`**: Per-attribute evaluation metrics for each trained model

## 🎯 Model Architectures Evaluated

| Model | Type | Key Characteristics |
|-------|------|--------------------|
| **ConvNeXt Tiny** | CNN | Modern ConvNet design, excellent performance |
| **EfficientNet B0** | CNN | Efficient scaling, compact model |
| **InceptionNext Tiny** | CNN | Inception-based architecture |
| **Swin Transformer** | Transformer | Hierarchical vision transformer |
| **ViT Base** | Transformer | Pure transformer architecture |
| **FocalNet Tiny SRF** | CNN | Focal modulation networks (CelebA) |

## 📝 Citation & Credits

**Datasets**: 
- ISGD: Custom dataset
- CelebA: Liu et al., "Deep Learning Face Attributes in the Wild" (ICCV 2015)

**Model Architectures**:
- ConvNeXt: Liu et al., "A ConvNet for the 2020s"
- Vision Transformer: Dosovitskiy et al., "An Image is Worth 16x16 Words"
- Swin Transformer: Liu et al., "Swin Transformer: Hierarchical Vision Transformer"
- EfficientNet: Tan & Le, "EfficientNet: Rethinking Model Scaling"
- FocalNet: Yang et al., "Focal Modulation Networks"

---

**Last Updated**: December 2024  
**Project Status**: Active Development