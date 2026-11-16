#!/bin/sh
set -e
cd /app

# If no arguments passed, use TRAIN_MODE env var to run train or eval
if [ $# -eq 0 ]; then
  exec python3 train.py --mode "$TRAIN_MODE" --episodes "$EPISODES" --max_steps "$MAX_STEPS"
else
  exec "$@"
fi
