#!/usr/bin/env python3
"""Скрипт для запуска обучения модели."""

import sys
import os

# Добавляем путь к модулю
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Теперь импортируем основной модуль
from fresh_rotten_classification.train import main

if __name__ == "__main__":
    main()
