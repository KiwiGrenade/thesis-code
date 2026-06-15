from torchvision.models import swin_v2_t
import torch.nn as nn


def build_model(num_classes: int):
    model = swin_v2_t(weights=None)

    model.head = nn.Linear(
        model.head.in_features,
        num_classes
    )

    return model


if __name__ == "__main__":
    model = build_model(num_classes=23)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(model)

    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")