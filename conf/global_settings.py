"""
configurations for this project
author Long-Chen Shen
"""
import os
from datetime import datetime


# directory to save weights file
CHECKPOINT_PATH = 'checkpoint'
CHECKPOINT_TRANSFER_PATH = 'checkpoint_transfer'

# total training epoches
EPOCH = 100
EPOCH_TRANSFORM = 20

PATIENCE = 10
PATIENCE_TRANSFORM = 6

MILESTONES = [6, 12, 18]
MILESTONES_TRANSFORM = [6, 12, 18]

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
# time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

# tensorboard log dir
LOG_DIR = 'runs'
LOG_TRANSFER_DIR = 'runs_transfer'

# save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 2

OUTPUT_INTERVAL = 200
OUTPUT_INTERVAL_TRANSFER = 50
