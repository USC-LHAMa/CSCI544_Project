#!/usr/bin/sh

# Create virtual environment and switch to it
python -m venv .
. ./bin/activate

# Install requiremed libraries
pip install torch
pip install tensorboardX
pip install -r ./transformers/requirements.txt
pip install -r ./transformers/requirements-dev.txt

# Install LHAMa customized Transformers package
python -m pip install -e ./transformers

# Install tmux for running long-running processes
sudo apt-get install tmux
