# Intel Image Classification (EfficientNet-B0)

A professional image classification project built using **PyTorch** and **EfficientNet-B0 (ImageNet pretrained)** on the **Intel Scene Classification dataset**.

This repository follows a clean, scalable architecture inspired by real-world deep learning projects and is suitable for freelance, production, and research use cases.

---

## 📌 Project Overview

- **Dataset:** Intel Image Classification (Kaggle)
- **Task:** Scene classification
- **Framework:** PyTorch
- **Model:** EfficientNet-B0 (pretrained on ImageNet)
- **Train / Validation Split:** 80 / 20
- **Evaluation Metrics:**
  - Accuracy
  - Confusion Matrix
  - Classification Report

---

## 📂 Dataset Structure

The project expects the following dataset structure (default Kaggle paths):
```
seg_train/seg_train/
seg_test/seg_test/
seg_pred/seg_pred/ (optional)

```

Dataset source:  
**Kaggle – puneet6060/intel-image-classification**

---

## 🧠 Model & Training Strategy

- EfficientNet-B0 with ImageNet pretrained weights
- Configurable fine-tuning modes:
  - `feature_extraction` (classifier only)
  - `fine_tune` (classifier + last feature blocks)
  - `full` (entire model)
- Loss Function: Cross Entropy Loss
- Optimizer: Adam
- Learning Rate Scheduler: ReduceLROnPlateau
- Early Stopping based on validation loss

---

## 🏗 Project Structure

```
intel-image-classification/
│
├── src/
│ ├── data.py # Data loading & transforms
│ ├── model.py # EfficientNet-B0 definition
│ ├── engine.py # Training & validation loops
│ └── train.py # Training orchestration script
│
├── notebooks/
│ └── intel_image_classification_full.ipynb # Self-contained notebook
│
├── configs/
│ └── config.yaml # Training & experiment configuration
│
├── requirements.txt
└── README.md
```


---

## 📓 Notebook

A **self-contained Jupyter Notebook** is provided:

```
notebooks/intel_image_classification_full.ipynb
```


✔ Includes **all code** (data, model, training, evaluation)  
✔ No dependency on `src/` files  
✔ Ideal for reviewers, clients, and demonstrations  

---

## 🚀 Training Script

The main training pipeline is implemented in:

```
src/train.py
```


It orchestrates:
- Data loading
- Model initialization
- Training & validation
- Checkpoint saving
- Evaluation & reporting

---

## 📊 Outputs

Training artifacts are automatically saved under:

```
outputs/
├── checkpoints/
│ └── efficientnet_b0_intel_best.pth
│
├── figures/
│ ├── training_curves.png
│ └── confusion_matrix.png
│
└── reports/
└── classification_report.txt
```

---

## ⚙ Configuration

All experiment settings are centralized in:

```
configs/config.yaml
```

This includes:
- Dataset paths
- Image preprocessing
- Training hyperparameters
- Scheduler settings
- Output directories
- Random seed

---

## 📦 Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```
Author
Mohamed Fathy


