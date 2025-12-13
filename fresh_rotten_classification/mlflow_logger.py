"""MLflow логгер для PyTorch Lightning."""

from typing import Any

import git
import mlflow
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import MLFlowLogger


class SimpleMLflowLogger(MLFlowLogger):
    """Упрощенный MLflow логгер."""

    def __init__(
        self,
        experiment_name: str = "fresh_rotten_classification",
        tracking_uri: str = "http://127.0.0.1:8080",
        run_name: str | None = None,
        **kwargs,
    ):
        super().__init__(
            experiment_name=experiment_name, tracking_uri=tracking_uri, run_name=run_name, **kwargs
        )

        # Логируем git commit id (требование задания)
        self._log_git_info()

    def _log_git_info(self):
        """Логирование git commit id (требование задания)."""
        try:
            repo = git.Repo(search_parent_directories=True)
            commit_hash = repo.head.object.hexsha
            self.experiment.set_tag(self.run_id, "git_commit", commit_hash)
        except Exception:
            # Если git не настроен, просто пропускаем
            pass

    def log_hyperparams(self, params: dict[str, Any]):
        """Логирование гиперпараметров (требование задания)."""
        if isinstance(params, DictConfig):
            params = OmegaConf.to_container(params, resolve=True)
        super().log_hyperparams(params)


class MLflowMetricsCallback(Callback):
    """Callback для логирования метрик в MLflow (минимум 3 графика)."""

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Логируем метрики в конце каждой эпохи."""
        metrics = trainer.callback_metrics

        # 1. Train Loss
        if "train_loss_epoch" in metrics:
            mlflow.log_metric("train_loss", metrics["train_loss_epoch"].item())

        # 2. Validation Loss
        if "val_loss" in metrics:
            mlflow.log_metric("val_loss", metrics["val_loss"].item())

        # 3. Train Accuracy
        if "train_acc_epoch" in metrics:
            mlflow.log_metric("train_acc", metrics["train_acc_epoch"].item())

        # 4. Validation Accuracy (дополнительно)
        if "val_acc" in metrics:
            mlflow.log_metric("val_acc", metrics["val_acc"].item())

        # 5. Learning Rate (дополнительный график)
        if "learning_rate" in metrics:
            mlflow.log_metric("learning_rate", metrics["learning_rate"].item())


def setup_mlflow(cfg: DictConfig) -> tuple:
    """
    Настройка MLflow.

    Returns:
        tuple: (logger, callbacks) или (None, []) если MLflow отключен
    """
    mlflow_config = cfg.get("mlflow", {})

    if not mlflow_config.get("enabled", True):
        return None, []

    try:
        # Логгер для гиперпараметров и git commit
        logger = SimpleMLflowLogger(
            experiment_name=mlflow_config.get("experiment_name", "fresh_rotten_classification"),
            tracking_uri=mlflow_config.get("tracking_uri", "http://127.0.0.1:8080"),
            run_name=mlflow_config.get("run_name", None),
        )

        # Callback для метрик (минимум 3 графика)
        metrics_callback = MLflowMetricsCallback()

        return logger, [metrics_callback]

    except Exception as e:
        print(f"Ошибка при настройке MLflow: {e}")
        print("Продолжаем без MLflow...")
        return None, []
