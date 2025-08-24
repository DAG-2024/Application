#!/bin/bash

export PYTHONPATH=/Users/gefenrashty/PycharmProjects/DAG/Application

# Start Stitcher
python3 -m src.stitcher.main &

# Start Controller
python3 src/controller/controller_run.py &

# Wait for both to finish (optional, keeps script running)
wait
