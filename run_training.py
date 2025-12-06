#!/usr/bin/env python
"""Запуск обучения с Hydra конфигом.
Примеры вызова:
    python run_training.py
    python run_training.py training.max_epochs=1
"""

# Этот файл теперь просто импортирует и запускает train.py
from fresh_rotten_classification.train import main

if __name__ == "__main__":
    main()
