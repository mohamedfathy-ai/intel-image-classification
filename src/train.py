# Intel Image Classification - Training Orchestrator
# Dataset: puneet6060/intel-image-classification
# Model: EfficientNet-B0 (ImageNet pretrained)

import os
import copy
from tqdm.auto import tqdm

import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from data import create_dataloaders
from model import EfficientNetB0
from engine import train_one_epoch, valid_one_epoch


# ================== DEVICE ==================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# ================== PATHS ==================
TRAIN_DIR = "/kaggle/input/intel-image-classification/seg_train/seg_train"
TEST_DIR  = "/kaggle/input/intel-image-classification/seg_test/seg_test"

OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ================== HYPERPARAMETERS ==================
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4
PATIENCE = 5
FINE_TUNE_MODE = "fine_tune"


# ================== DATA ==================
train_loader, val_loader, test_loader, classes = create_dataloaders(
    train_dir=TRAIN_DIR,
    test_dir=TEST_DIR,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE
)

num_classes = len(classes)
print("Classes:", classes)


# ================== MODEL ==================
model = EfficientNetB0(
    num_classes=num_classes,
    mode=FINE_TUNE_MODE
).to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode="min",
    factor=0.1,
    patience=2,
    min_lr=1e-7
)


# ================== TRAINING ==================
best_val_loss = float("inf")
best_weights = copy.deepcopy(model.state_dict())
no_improve = 0

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

    train_loss, train_acc = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device
    )

    val_loss, val_acc = valid_one_epoch(
        model=model,
        loader=val_loader,
        loss_fn=loss_fn,
        device=device
    )

    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        print("Validation loss improved. Saving model.")
        best_val_loss = val_loss
        best_weights = copy.deepcopy(model.state_dict())
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break


# ================== SAVE BEST MODEL ==================
model.load_state_dict(best_weights)
checkpoint_path = os.path.join(CHECKPOINT_DIR, "efficientnet_b0_intel_best.pth")
torch.save(model.state_dict(), checkpoint_path)
print(f"Best model saved to: {checkpoint_path}")


# ================== PLOTS ==================
epochs = range(1, len(history["train_loss"]) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, history["train_loss"], label="Train Loss")
plt.plot(epochs, history["val_loss"], label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, history["train_acc"], label="Train Acc")
plt.plot(epochs, history["val_acc"], label="Val Acc")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "training_curves.png"))
plt.close()


# ================== TEST & EVALUATION ==================
model.eval()
y_true, y_pred = [], []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=classes)

plt.figure(figsize=(6, 5))
disp.plot(cmap="Blues", xticks_rotation=45, colorbar=False)
plt.title("Confusion Matrix - Intel Test Set")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"))
plt.close()

report = classification_report(y_true, y_pred, target_names=classes)
with open(os.path.join(REPORTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

print("\nClassification Report:\n")
print(report)
