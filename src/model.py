import torch
from torch import nn
from torchvision import models


def set_trainable(models_list, requires_grad: bool):
    if not isinstance(models_list, (list, tuple)):
        models_list = [models_list]
    for model in models_list:
        for p in model.parameters():
            p.requires_grad = requires_grad


def EfficientNetB0(num_classes: int, mode: str = "fine_tune") -> nn.Module:
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if mode == "feature_extraction":
        set_trainable(model, False)
        set_trainable(model.classifier[1], True)

    elif mode == "fine_tune":
        set_trainable(model, False)
        set_trainable(
            [model.classifier[1], model.features[-1], model.features[-2]],
            True
        )

    elif mode == "full":
        set_trainable(model, True)

    return model
