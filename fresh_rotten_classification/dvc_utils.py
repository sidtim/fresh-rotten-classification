"""Минимальные утилиты для работы с DVC."""

import subprocess
from pathlib import Path

from loguru import logger


def check_and_download_data(data_dir: str = "./data") -> bool:
    """
    Проверяет наличие данных и скачивает через DVC если их нет.

    Args:
        data_dir: Директория для данных

    Returns:
        bool: True если данные доступны
    """
    data_path = Path(data_dir)

    # Check data
    if data_path.exists() and any(data_path.iterdir()):
        logger.info(f"Data already exist in {data_dir}")
        return True

    # If no data than download via dvc
    logger.info(f"Download into {data_dir}...")

    # MakeDir if it need
    data_path.mkdir(parents=True, exist_ok=True)

    # dvc pull
    try:
        result = subprocess.run(
            ["poetry", "run", "dvc", "pull"],
            capture_output=True,
            text=True,
            cwd=".",  # pull from the root
        )

        if result.returncode == 0:
            logger.info("✅ Data download")
            return True
        else:  # noqa: RET505
            logger.error(f"❌ ERROR in DVC: {result.stderr}")
            return False

    except FileNotFoundError:
        logger.error("DVC doesn't download: poetry add dvc")
        return False
    except Exception as e:
        logger.error(f"Error : {e}")
        return False
