#!/bin/sh
set -e
cd /app

# Run MORL training by default; allow an explicit command to override it.
if [ $# -eq 0 ]; then
  exec python3 train_morl.py
else
  exec "$@"
fi
