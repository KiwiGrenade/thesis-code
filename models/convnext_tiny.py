from torchvision.models import convnext_tiny
import torch.nn as nn


def build_model(num_classes: int):
    model = convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features,
        num_classes
    )
    return model