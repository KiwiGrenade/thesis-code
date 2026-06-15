from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import json
import re

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_HISTORY_FILE = "history_by_epoch.csv"
DEFAULT_TEST_SUMMARY_FILE = "test_summary.csv"
DEFAULT_BEST_VALIDATION_FILE = "best_validation_metrics.csv"
DEFAULT_CLASS_REPORT_FILE = "test_classification_report.csv"
DEFAULT_CONFUSION_MATRIX_FILE = "test_confusion_matrix.csv"


def _safe_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^\w\-.]+", "_", name, flags=re.UNICODE)
    return name.strip("_") or "plot"


def _experiment_name_from_dir(experiment_dir: Path) -> str:
    config_path = experiment_dir / "config.json"

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            for key in ["session_name", "experiment_name", "model_name", "model"]:
                value = config.get(key)
                if value:
                    return str(value)
        except Exception:
            pass

    return experiment_dir.name


def _find_metrics_dir(experiment_dir: Path) -> Path:
    experiment_dir = Path(experiment_dir)

    direct = experiment_dir / "metrics"
    if direct.exists():
        return direct

    if experiment_dir.name == "metrics" and experiment_dir.exists():
        return experiment_dir

    return direct


def load_history(
    experiment_dir: str | Path,
    experiment_name: str | None = None,
    history_file: str = DEFAULT_HISTORY_FILE,
) -> pd.DataFrame:
    experiment_dir = Path(experiment_dir)
    metrics_dir = _find_metrics_dir(experiment_dir)
    history_path = metrics_dir / history_file

    if not history_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku historii: {history_path}")

    df = pd.read_csv(history_path)

    if "epoch" not in df.columns:
        raise ValueError(f"Plik {history_path} nie zawiera kolumny 'epoch'.")

    df["epoch"] = df["epoch"].astype(int)
    df["experiment"] = experiment_name or _experiment_name_from_dir(experiment_dir)
    df["experiment_dir"] = str(experiment_dir)

    return df


def load_histories(
    experiment_dirs: Iterable[str | Path],
    experiment_names: Sequence[str] | None = None,
    history_file: str = DEFAULT_HISTORY_FILE,
) -> pd.DataFrame:
    experiment_dirs = list(experiment_dirs)

    if experiment_names is not None and len(experiment_names) != len(experiment_dirs):
        raise ValueError("Liczba nazw eksperymentów musi być taka sama jak liczba folderów.")

    frames = []

    for idx, experiment_dir in enumerate(experiment_dirs):
        name = experiment_names[idx] if experiment_names is not None else None
        frames.append(
            load_history(
                experiment_dir=experiment_dir,
                experiment_name=name,
                history_file=history_file,
            )
        )

    if not frames:
        raise ValueError("Nie podano żadnego folderu z wynikami.")

    return pd.concat(frames, ignore_index=True)


def load_test_summaries(
    experiment_dirs: Iterable[str | Path],
    experiment_names: Sequence[str] | None = None,
    test_summary_file: str = DEFAULT_TEST_SUMMARY_FILE,
) -> pd.DataFrame:
    experiment_dirs = list(experiment_dirs)

    if experiment_names is not None and len(experiment_names) != len(experiment_dirs):
        raise ValueError("Liczba nazw eksperymentów musi być taka sama jak liczba folderów.")

    frames = []

    for idx, experiment_dir in enumerate(experiment_dirs):
        experiment_dir = Path(experiment_dir)
        metrics_dir = _find_metrics_dir(experiment_dir)
        path = metrics_dir / test_summary_file

        if not path.exists():
            continue

        df = pd.read_csv(path)
        df["experiment"] = (
            experiment_names[idx]
            if experiment_names is not None
            else _experiment_name_from_dir(experiment_dir)
        )
        df["experiment_dir"] = str(experiment_dir)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    cols = ["experiment", "experiment_dir"]
    other_cols = [c for c in pd.concat(frames, ignore_index=True).columns if c not in cols]
    return pd.concat(frames, ignore_index=True)[cols + other_cols]


def load_best_validation(
    experiment_dirs: Iterable[str | Path],
    experiment_names: Sequence[str] | None = None,
    best_validation_file: str = DEFAULT_BEST_VALIDATION_FILE,
) -> pd.DataFrame:
    experiment_dirs = list(experiment_dirs)

    if experiment_names is not None and len(experiment_names) != len(experiment_dirs):
        raise ValueError("Liczba nazw eksperymentów musi być taka sama jak liczba folderów.")

    frames = []

    for idx, experiment_dir in enumerate(experiment_dirs):
        experiment_dir = Path(experiment_dir)
        metrics_dir = _find_metrics_dir(experiment_dir)
        path = metrics_dir / best_validation_file

        if not path.exists():
            continue

        df = pd.read_csv(path)
        df["experiment"] = (
            experiment_names[idx]
            if experiment_names is not None
            else _experiment_name_from_dir(experiment_dir)
        )
        df["experiment_dir"] = str(experiment_dir)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    cols = ["experiment", "experiment_dir"]
    all_df = pd.concat(frames, ignore_index=True)
    other_cols = [c for c in all_df.columns if c not in cols]
    return all_df[cols + other_cols]


def available_history_metrics(history_df: pd.DataFrame) -> list[str]:
    excluded = {"epoch", "step", "experiment", "experiment_dir"}
    metrics = []

    for col in history_df.columns:
        if col not in excluded and pd.api.types.is_numeric_dtype(history_df[col]):
            metrics.append(col)

    return metrics


def plot_metric(
    history_df: pd.DataFrame,
    metric: str,
    output_dir: str | Path | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    show: bool = True,
    save: bool = True,
    dpi: int = 160,
) -> Path | None:
    if metric not in history_df.columns:
        raise ValueError(f"Brak metryki '{metric}' w danych.")

    fig, ax = plt.subplots(figsize=(10, 5))

    for experiment, group in history_df.groupby("experiment", sort=False):
        group = group.sort_values("epoch")
        ax.plot(group["epoch"], group[metric], marker="o", linewidth=1.8, markersize=3, label=experiment)

    ax.set_xlabel("Epoka")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or metric)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = None

    if save:
        if output_dir is None:
            output_dir = Path("plots")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{_safe_name(metric)}.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path

def plot_metric_no_title(
    history_df: pd.DataFrame,
    metric: str,
    output_dir: str | Path | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    show: bool = True,
    save: bool = True,
    dpi: int = 160,
) -> Path | None:
    if metric not in history_df.columns:
        raise ValueError(f"Brak metryki '{metric}' w danych.")

    fig, ax = plt.subplots(figsize=(10, 5))

    for experiment, group in history_df.groupby("experiment", sort=False):
        group = group.sort_values("epoch")
        ax.plot(group["epoch"], group[metric], marker="o", linewidth=1.8, markersize=3, label=experiment)

    ax.set_xlabel("Epoka")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = None

    if save:
        if output_dir is None:
            output_dir = Path("plots")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{_safe_name(metric)}.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_train_val_pair(
    history_df: pd.DataFrame,
    train_metric: str,
    val_metric: str,
    output_dir: str | Path | None = None,
    title: str | None = None,
    show: bool = True,
    save: bool = True,
    dpi: int = 160,
) -> Path | None:
    missing = [m for m in [train_metric, val_metric] if m not in history_df.columns]
    if missing:
        raise ValueError(f"Brakuje kolumn: {missing}")

    fig, ax = plt.subplots(figsize=(10, 5))

    for experiment, group in history_df.groupby("experiment", sort=False):
        group = group.sort_values("epoch")
        ax.plot(
            group["epoch"],
            group[train_metric],
            linestyle="-",
            linewidth=1.8,
            label=f"{experiment} | train",
        )
        ax.plot(
            group["epoch"],
            group[val_metric],
            linestyle="--",
            linewidth=1.8,
            label=f"{experiment} | val",
        )

    ax.set_xlabel("Epoka")
    ax.set_ylabel(train_metric.replace("train_", ""))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = None

    if save:
        if output_dir is None:
            output_dir = Path("plots")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        name = f"{_safe_name(train_metric)}__{_safe_name(val_metric)}.png"
        output_path = output_dir / name
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path

def plot_metric_grid(
    history_df: pd.DataFrame,
    metrics: Sequence[str],
    output_dir: str | Path | None = None,
    show: bool = True,
    save: bool = True,
    dpi: int = 160,
) -> list[Path]:
    paths = []

    for metric in metrics:
        if metric in history_df.columns:
            path = plot_metric(
                history_df=history_df,
                metric=metric,
                output_dir=output_dir,
                show=show,
                save=save,
                dpi=dpi,
            )
            if path is not None:
                paths.append(path)

    return paths


def plot_test_metric_bar(
    test_df: pd.DataFrame,
    metric: str,
    output_dir: str | Path | None = None,
    title: str | None = None,
    show: bool = True,
    save: bool = True,
    dpi: int = 160,
) -> Path | None:
    if test_df.empty:
        raise ValueError("Brak danych testowych.")

    if metric not in test_df.columns:
        raise ValueError(f"Brak metryki '{metric}' w danych testowych.")

    plot_df = test_df[["experiment", metric]].dropna().copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df["experiment"], plot_df[metric])
    ax.set_xlabel("Eksperyment")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Test: {metric}")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = None

    if save:
        if output_dir is None:
            output_dir = Path("plots")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"test_{_safe_name(metric)}.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_confusion_matrix(
    confusion_matrix_csv: str | Path,
    output_path: str | Path | None = None,
    title: str = "Macierz pomyłek",
    normalize: bool = False,
    show_values: bool = False,
    show: bool = True,
    dpi: int = 180,
) -> Path | None:
    cm_df = pd.read_csv(confusion_matrix_csv, index_col=0)

    data = cm_df.astype(float)

    if normalize:
        row_sums = data.sum(axis=1).replace(0, 1)
        data = data.div(row_sums, axis=0)

    fig_size = max(8, min(18, 0.45 * len(data.columns)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(data.values, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(data.columns)))
    ax.set_yticks(range(len(data.index)))
    ax.set_xticklabels(data.columns, rotation=90)
    ax.set_yticklabels(data.index)
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Klasa rzeczywista")
    ax.set_title(title + (" — znormalizowana" if normalize else ""))

    if show_values:
        fmt = ".2f" if normalize else ".0f"
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, format(data.iloc[i, j], fmt), ha="center", va="center", fontsize=7)

    fig.tight_layout()

    saved_path = None

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        saved_path = output_path

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path
