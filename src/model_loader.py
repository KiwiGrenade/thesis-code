from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

import torch.nn as nn


def load_model_factory(model_file: str | Path, factory_name: str = "build_model") -> Callable[..., nn.Module]:
    model_file = Path(model_file)

    if not model_file.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku z modelem: {model_file}\n"
            f"Utwórz plik albo zmień MODEL_FILE w konfiguracji."
        )

    module_name = model_file.stem
    spec = importlib.util.spec_from_file_location(module_name, model_file)

    if spec is None or spec.loader is None:
        raise ImportError(f"Nie można załadować modułu z pliku: {model_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, factory_name):
        raise AttributeError(
            f"Plik {model_file} nie zawiera funkcji {factory_name}(num_classes)."
        )

    return getattr(module, factory_name)
