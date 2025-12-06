#!/usr/bin/env python
"""Скрипт для запуска обучения модели."""

import sys
from pathlib import Path

# Добавляем путь к модулю
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

# Теперь импортируем основной модуль
from fresh_rotten_classification.train import main

if __name__ == "__main__":
    main()
