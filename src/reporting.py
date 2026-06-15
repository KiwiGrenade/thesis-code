from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


HISTORY_COLUMNS = [
    "epoch",

    "train_loss",
    "train_acc",
    "train_f1",
    "train_precision",
    "train_recall",
    "train_macro_f1",
    "train_macro_precision",
    "train_macro_recall",
    "train_balanced_acc",

    "val_loss",
    "val_acc",
    "val_f1",
    "val_precision",
    "val_recall",
    "val_macro_f1",
    "val_macro_precision",
    "val_macro_recall",
    "val_balanced_acc",

    "lr",
    "lr-AdamW",
]

TEST_SUMMARY_COLUMNS = [
    "test_loss",
    "test_acc",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_macro_f1",
    "test_macro_precision",
    "test_macro_recall",
    "test_balanced_acc",
]


def _as_dataframe_row(data: Any) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()

    if isinstance(data, list):
        return pd.DataFrame(data)

    if isinstance(data, dict):
        return pd.DataFrame([data])

    return pd.DataFrame(data)


def _sort_columns(df: pd.DataFrame, preferred_order: list[str]) -> pd.DataFrame:
    existing_cols = [c for c in preferred_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_cols]
    return df[existing_cols + remaining_cols]


def _make_history_by_epoch(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if "epoch" not in metrics_df.columns:
        raise ValueError("Plik metrics.csv nie zawiera kolumny 'epoch'.")

    history_df = (
        metrics_df
        .dropna(subset=["epoch"])
        .groupby("epoch", as_index=False)
        .first()
    )

    history_df["epoch"] = history_df["epoch"].astype(int) + 1
    history_df = _sort_columns(history_df, HISTORY_COLUMNS)

    return history_df


def save_training_artifacts(
    csv_log_dir: str | Path,
    metrics_dir: str | Path,
    test_results: list[dict[str, Any]] | dict[str, Any] | None = None,
    y_true: Any | None = None,
    y_pred: Any | None = None,
    class_names: list[str] | None = None,
    monitor_metric: str = "val_macro_f1",
) -> pd.DataFrame:
    csv_log_dir = Path(csv_log_dir)
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = csv_log_dir / "metrics.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku metrics.csv: {metrics_path}")

    metrics_df = pd.read_csv(metrics_path)

    history_df = _make_history_by_epoch(metrics_df)
    history_df.to_csv(metrics_dir / "history_by_epoch.csv", index=False)

    if monitor_metric in history_df.columns and not history_df[monitor_metric].dropna().empty:
        best_idx = history_df[monitor_metric].idxmax()
        best_validation_df = history_df.loc[[best_idx]]
        best_validation_df.to_csv(
            metrics_dir / "best_validation_metrics.csv",
            index=False,
        )

    if test_results is not None:
        test_summary_df = _as_dataframe_row(test_results)
        test_summary_df = _sort_columns(test_summary_df, TEST_SUMMARY_COLUMNS)
        test_summary_df.to_csv(
            metrics_dir / "test_summary.csv",
            index=False,
        )

    if y_true is not None and y_pred is not None and class_names is not None:
        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(metrics_dir / "test_classification_report.csv")

        cm = confusion_matrix(y_true, y_pred)

        cm_df = pd.DataFrame(
            cm,
            index=class_names,
            columns=class_names,
        )
        cm_df.to_csv(metrics_dir / "test_confusion_matrix.csv")

        per_class_metrics_df = report_df.loc[
            class_names,
            ["precision", "recall", "f1-score", "support"],
        ]
        per_class_metrics_df.to_csv(metrics_dir / "test_per_class_metrics.csv")

        per_class_recall_df = report_df.loc[
            class_names,
            ["recall", "support"],
        ]
        per_class_recall_df.to_csv(metrics_dir / "per_class_recall.csv")

    return history_df
