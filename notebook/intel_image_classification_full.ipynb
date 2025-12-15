import os
import copy
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Kaggle paths
train_dir = "/kaggle/input/intel-image-classification/seg_train/seg_train"
test_dir  = "/kaggle/input/intel-image-classification/seg_test/seg_test"
pred_dir  = "/kaggle/input/intel-image-classification/seg_pred/seg_pred"  # optional

# Outputs
OUTPUT_DIR = "outputs"
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
FIG_DIR  = os.path.join(OUTPUT_DIR, "figures")
REP_DIR  = os.path.join(OUTPUT_DIR, "reports")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REP_DIR, exist_ok=True)

img_size = 224
batch_size = 32
num_epochs = 10
lr = 1e-4
patience = 5
mode = "fine_tune"  # feature_extraction | fine_tune | full

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

def denormalize(tensor, mean, std):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def accuracy_fn(y_true, y_pred):
    return (y_true == y_pred).sum().item()

def show_random_grid(dataset, classes, title="train", row=3, col=4, seed=42):
    torch.manual_seed(seed)
    fig = plt.figure(figsize=(12, 9))

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    for i in range(1, row * col + 1):
        idx = torch.randint(0, len(dataset), size=[1]).item()
        img_ten, label = dataset[idx]
        img_denorm = denormalize(img_ten, mean, std).clamp(0, 1)
        img = img_denorm.permute(1, 2, 0)

        fig.add_subplot(row, col, i)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{title} - {classes[label]}")
    plt.tight_layout()
    plt.show()

full_train_data = datasets.ImageFolder(root=train_dir, transform=transform)
classes = full_train_data.classes
num_classes = len(classes)

train_size = int(0.8 * len(full_train_data))
val_size   = len(full_train_data) - train_size
train_data, val_data = random_split(full_train_data, [train_size, val_size])

test_data = datasets.ImageFolder(root=test_dir, transform=transform)

print("Classes:", classes)
print("Num classes:", num_classes)
print("Train samples:", len(train_data))
print("Val samples:", len(val_data))
print("Test samples:", len(test_data))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

show_random_grid(train_data, classes, title="train")
show_random_grid(test_data,  classes, title="test")

def set_trainable(models_list, requires_grad: bool):
    if not isinstance(models_list, (list, tuple)):
        models_list = [models_list]
    for model in models_list:
        for p in model.parameters():
            p.requires_grad = requires_grad

def EfficientNetB0(num_classes: int, mode: str = "fine_tune") -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if mode == "feature_extraction":
        set_trainable(model, False)
        set_trainable(model.classifier[1], True)

    elif mode == "fine_tune":
        set_trainable(model, False)
        set_trainable([model.classifier[1], model.features[-1], model.features[-2]], True)

    elif mode == "full":
        set_trainable(model, True)

    else:
        raise ValueError("Unknown mode. Use: feature_extraction | fine_tune | full")

    return model

model = EfficientNetB0(num_classes=num_classes, mode=mode).to(device)
print("Model ready on device.")

def train_one_epoch(model, data_loader, optimizer, loss_fn, device):
    tr_loss, tr_acc = 0.0, 0.0
    total_samples = 0
    model.train()

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        bs = images.size(0)
        total_samples += bs

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        tr_loss += loss.item() * bs
        preds = outputs.argmax(dim=1)
        tr_acc += accuracy_fn(labels, preds)

        loss.backward()
        optimizer.step()

    return tr_loss / total_samples, tr_acc / total_samples


def valid_one_epoch(model, data_loader, loss_fn, device):
    val_loss, val_acc = 0.0, 0.0
    total_samples = 0
    model.eval()

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            bs = images.size(0)
            total_samples += bs

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            val_loss += loss.item() * bs
            preds = outputs.argmax(dim=1)
            val_acc += accuracy_fn(labels, preds)

    return val_loss / total_samples, val_acc / total_samples

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr
)
loss_fn = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode="min",
    factor=0.1,
    patience=2,
    min_lr=1e-7
)

best_val_loss = float("inf")
best_wts = copy.deepcopy(model.state_dict())
no_improve = 0

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in tqdm(range(1, num_epochs + 1)):
    print(f"\nEpoch {epoch}/{num_epochs}")

    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
    v_loss, v_acc   = valid_one_epoch(model, val_loader, loss_fn, device)

    scheduler.step(v_loss)

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(v_loss)
    history["val_acc"].append(v_acc)

    print(f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")
    print(f"Val   Loss: {v_loss:.4f} | Val   Acc: {v_acc:.4f}")

    if v_loss < best_val_loss:
        print("Validation loss improved. Saving best weights.")
        best_val_loss = v_loss
        best_wts = copy.deepcopy(model.state_dict())
        no_improve = 0
    else:
        no_improve += 1
        print(f"No improvement for {no_improve}/{patience}")
        if no_improve >= patience:
            print("Early stopping triggered!")
            break

model.load_state_dict(best_wts)
ckpt_path = os.path.join(CKPT_DIR, "efficientnet_b0_intel_best.pth")
torch.save(model.state_dict(), ckpt_path)
print("Best model saved to:", ckpt_path)

epochs = range(1, len(history["train_loss"]) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
plt.plot(epochs, history["val_loss"],   marker="o", label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Train vs Val Loss")
plt.grid(True, linestyle="--", alpha=0.4); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, history["train_acc"], marker="o", label="Train Acc")
plt.plot(epochs, history["val_acc"],   marker="o", label="Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Train vs Val Accuracy")
plt.grid(True, linestyle="--", alpha=0.4); plt.legend()

plt.tight_layout()
curve_path = os.path.join(FIG_DIR, "training_curves.png")
plt.savefig(curve_path)
plt.show()
print("Saved:", curve_path)

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

plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(xticks_rotation=45, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix - Intel Test Set")
plt.tight_layout()

cm_path = os.path.join(FIG_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.show()
print("Saved:", cm_path)

report = classification_report(y_true, y_pred, target_names=classes)
print("\nClassification Report:\n")
print(report)

rep_path = os.path.join(REP_DIR, "classification_report.txt")
with open(rep_path, "w") as f:
    f.write(report)
print("Saved:", rep_path)

