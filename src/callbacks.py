from lightning.pytorch.callbacks import Callback
import pandas as pd


class EpochMetricsLogger(Callback):
    def __init__(self, save_path):
        self.save_path = save_path
        self.rows = []

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        row = {
            "epoch": trainer.current_epoch + 1,
            "train_loss": self._get(metrics, "train_loss"),
            "train_acc": self._get(metrics, "train_acc"),
            "train_macro_f1": self._get(metrics, "train_macro_f1"),
            "val_loss": self._get(metrics, "val_loss"),
            "val_acc": self._get(metrics, "val_acc"),
            "val_macro_f1": self._get(metrics, "val_macro_f1"),
            "val_macro_precision": self._get(metrics, "val_macro_precision"),
            "val_macro_recall": self._get(metrics, "val_macro_recall"),
        }

        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.save_path, index=False)

        print(
            f"Epoch {row['epoch']:03d} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"train_acc={row['train_acc']:.4f} | "
            f"train_f1={row['train_macro_f1']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"val_acc={row['val_acc']:.4f} | "
            f"val_f1={row['val_macro_f1']:.4f}"
        )

    def _get(self, metrics, name):
        value = metrics.get(name)
        if value is None:
            return float("nan")
        if hasattr(value, "item"):
            return value.item()
        return value