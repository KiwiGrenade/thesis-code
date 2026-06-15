from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


class LitImageClassifier(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        num_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_name: str = "adamw",
        scheduler_name: str = "cosine",
        max_epochs: int = 50,
        class_names: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__()

        self.model = model
        self.criterion = criterion
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.max_epochs = max_epochs
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.config = config or {}

        self.save_hyperparameters(ignore=["model", "criterion"])

        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="micro")
        self.train_precision = MulticlassPrecision(num_classes=num_classes, average="micro")
        self.train_recall = MulticlassRecall(num_classes=num_classes, average="micro")
        self.train_macro_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.train_macro_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.train_macro_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.train_balanced_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")

        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="micro")
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average="micro")
        self.val_recall = MulticlassRecall(num_classes=num_classes, average="micro")
        self.val_macro_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_macro_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.val_macro_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.val_balanced_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_per_class_recall = MulticlassRecall(num_classes=num_classes, average=None)

        self.test_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="micro")
        self.test_precision = MulticlassPrecision(num_classes=num_classes, average="micro")
        self.test_recall = MulticlassRecall(num_classes=num_classes, average="micro")
        self.test_macro_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_macro_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.test_macro_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.test_balanced_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")

        self.test_per_class_precision = MulticlassPrecision(num_classes=num_classes, average=None)
        self.test_per_class_recall = MulticlassRecall(num_classes=num_classes, average=None)
        self.test_per_class_f1 = MulticlassF1Score(num_classes=num_classes, average=None)
        self.test_confmat = MulticlassConfusionMatrix(num_classes=num_classes)

        self.test_targets: list[torch.Tensor] | torch.Tensor = []
        self.test_preds: list[torch.Tensor] | torch.Tensor = []
        self.test_logits: list[torch.Tensor] | torch.Tensor = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _log_stage_loss(self, stage: str, loss: torch.Tensor) -> None:
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=(stage in {"train", "val"}),
            sync_dist=True,
        )

    def _log_metric(
        self,
        name: str,
        metric_value: torch.Tensor,
        *,
        prog_bar: bool = False,
    ) -> None:
        self.log(
            name,
            metric_value,
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar,
            sync_dist=True,
        )

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        self._log_stage_loss(stage, loss)

        if stage == "train":
            self._log_train_metrics(preds, labels)
        elif stage == "val":
            self._log_val_metrics(preds, labels)
        elif stage == "test":
            self._log_test_metrics(preds, labels)
            self.test_targets.append(labels.detach().cpu())
            self.test_preds.append(preds.detach().cpu())
            self.test_logits.append(logits.detach().cpu())
        else:
            raise ValueError(f"Nieznany etap: {stage}")

        return loss

    def _log_train_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        self._log_metric("train_acc", self.train_acc(preds, labels), prog_bar=True)
        self._log_metric("train_f1", self.train_f1(preds, labels))
        self._log_metric("train_precision", self.train_precision(preds, labels))
        self._log_metric("train_recall", self.train_recall(preds, labels))
        self._log_metric("train_macro_f1", self.train_macro_f1(preds, labels), prog_bar=True)
        self._log_metric("train_macro_precision", self.train_macro_precision(preds, labels))
        self._log_metric("train_macro_recall", self.train_macro_recall(preds, labels))
        self._log_metric("train_balanced_acc", self.train_balanced_acc(preds, labels))

    def _log_val_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        self._log_metric("val_acc", self.val_acc(preds, labels), prog_bar=True)
        self._log_metric("val_f1", self.val_f1(preds, labels))
        self._log_metric("val_precision", self.val_precision(preds, labels))
        self._log_metric("val_recall", self.val_recall(preds, labels))
        self._log_metric("val_macro_f1", self.val_macro_f1(preds, labels), prog_bar=True)
        self._log_metric("val_macro_precision", self.val_macro_precision(preds, labels))
        self._log_metric("val_macro_recall", self.val_macro_recall(preds, labels))
        self._log_metric("val_balanced_acc", self.val_balanced_acc(preds, labels))
        self.val_per_class_recall.update(preds, labels)

    def _log_test_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        self._log_metric("test_acc", self.test_acc(preds, labels), prog_bar=True)
        self._log_metric("test_f1", self.test_f1(preds, labels))
        self._log_metric("test_precision", self.test_precision(preds, labels))
        self._log_metric("test_recall", self.test_recall(preds, labels))
        self._log_metric("test_macro_f1", self.test_macro_f1(preds, labels), prog_bar=True)
        self._log_metric("test_macro_precision", self.test_macro_precision(preds, labels))
        self._log_metric("test_macro_recall", self.test_macro_recall(preds, labels))
        self._log_metric("test_balanced_acc", self.test_balanced_acc(preds, labels))

        self.test_per_class_precision.update(preds, labels)
        self.test_per_class_recall.update(preds, labels)
        self.test_per_class_f1.update(preds, labels)
        self.test_confmat.update(preds, labels)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def on_test_epoch_start(self) -> None:
        self.test_targets = []
        self.test_preds = []
        self.test_logits = []

    def on_test_epoch_end(self) -> None:
        if self.test_targets:
            self.test_targets = torch.cat(self.test_targets).numpy()
            self.test_preds = torch.cat(self.test_preds).numpy()
            self.test_logits = torch.cat(self.test_logits).numpy()

    def configure_optimizers(self):
        optimizer_name = self.optimizer_name.lower().strip()

        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Nieznany optimizer: {self.optimizer_name}")

        scheduler_name = self.scheduler_name.lower().strip()

        if scheduler_name in {"none", ""}:
            return optimizer

        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "name": "lr",
                },
            }

        if scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(1, self.max_epochs // 3),
                gamma=0.1,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "name": "lr",
                },
            }

        raise ValueError(f"Nieznany scheduler: {self.scheduler_name}")
