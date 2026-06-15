from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch

def collect_model_statistics(
    model,
    experiment_name: str,
    image_size: int = 256,
    batch_sizes: list[int] = [1, 16, 32],
    device: str = "cuda",
):
    model = model.to(device)
    model.eval()

    results = {}

    # =========================
    # PARAMS
    # =========================

    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    non_trainable_params = total_params - trainable_params

    results["total_params"] = total_params
    results["trainable_params"] = trainable_params
    results["non_trainable_params"] = non_trainable_params

    # =========================
    # MODEL SIZE
    # =========================

    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    model_size_mb = (param_size + buffer_size) / 1024**2

    results["model_size_mb"] = model_size_mb


    # =========================
    # INFERENCE
    # =========================

    for batch_size in batch_sizes:

        x = torch.randn(
            batch_size,
            3,
            image_size,
            image_size
        ).to(device)

        # warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(100):
                _ = model(x)

        if device == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        total_time = end - start

        avg_batch_time = total_time / 100
        avg_image_time = avg_batch_time / batch_size

        results[f"bs{batch_size}_batch_time_ms"] = avg_batch_time * 1000
        results[f"bs{batch_size}_image_time_ms"] = avg_image_time * 1000

        results[f"bs{batch_size}_images_per_second"] = (
            batch_size / avg_batch_time
        )

    # =========================
    # VRAM
    # =========================

    if device == "cuda":

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        x = torch.randn(
            32,
            3,
            image_size,
            image_size
        ).to(device)

        with torch.no_grad():
            _ = model(x)

        torch.cuda.synchronize()

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

        results["peak_vram_mb"] = peak_memory

    else:
        results["peak_vram_mb"] = None

    results["experiment_name"] = experiment_name
    results["image_size"] = image_size

    return results

def save_model_statistics(stats, output_csv):

    df = pd.DataFrame([stats])

    output_csv = Path(output_csv)

    if output_csv.exists():
        old_df = pd.read_csv(output_csv)
        df = pd.concat([old_df, df], ignore_index=True)

    df.to_csv(output_csv, index=False)

    return df