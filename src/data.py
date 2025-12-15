from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_transforms(img_size=224):
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])


def create_dataloaders(train_dir, test_dir, batch_size=32, img_size=224, num_workers=2):
    transform = get_transforms(img_size)

    full_train_data = datasets.ImageFolder(train_dir, transform=transform)
    classes = full_train_data.classes

    train_size = int(0.8 * len(full_train_data))
    val_size   = len(full_train_data) - train_size
    train_data, val_data = random_split(full_train_data, [train_size, val_size])

    test_data = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, classes
