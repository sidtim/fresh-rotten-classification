#!/usr/bin/env python
"""Run training with Hydra configuration.
Usage examples:
    python run_training.py
    python run_training.py model=simple_cnn
    python run_training.py data.batch_size=64 training.learning_rate=0.0001
    python run_training.py data.use_augmentation=false training.gpu=true
    python run_training.py -m data.batch_size=32,64,128
"""

# Этот файл теперь просто импортирует и запускает train.py
from fresh_rotten_classification.train import main

if __name__ == "__main__":
    main()
