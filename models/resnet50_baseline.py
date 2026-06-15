import torch.nn as nn
from torchvision.models import resnet50


def build_model(num_classes: int):
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
