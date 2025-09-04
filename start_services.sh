#!/bin/bash

export PYTHONPATH=/Users/gefenrashty/PycharmProjects/DAG/Application

# Start Stitcher
python -m src.stitcher.main &

# Start Controller
python src/controller/controller_run.py &

# Wait for both to finish (optional, keeps script running)
wait
