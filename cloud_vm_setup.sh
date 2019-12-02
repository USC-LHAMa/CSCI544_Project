#!/usr/bin/bash

# Create virtual environment and switch to it
python -m venv .
source ./bin/activate

# Install requiremed libraries
pip install torch
pip install tensorboardX
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install LHAMa customized Transformers package
python -m pip install -e .

# Install tmux for running long-running processes
sudo apt-get install tmux
