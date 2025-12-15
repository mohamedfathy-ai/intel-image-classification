import torch


def accuracy_fn(y_true, y_pred):
    return (y_true == y_pred).sum().item()


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, total_acc, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        bs = images.size(0)
        total += bs

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        preds = outputs.argmax(dim=1)
        total_acc += accuracy_fn(labels, preds)
        total_loss += loss.item() * bs

        loss.backward()
        optimizer.step()

    return total_loss / total, total_acc / total


def valid_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_acc, total = 0, 0, 0

    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            bs = images.size(0)
            total += bs

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            preds = outputs.argmax(dim=1)
            total_acc += accuracy_fn(labels, preds)
            total_loss += loss.item() * bs

    return total_loss / total, total_acc / total
