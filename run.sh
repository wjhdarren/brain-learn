#!/bin/bash

# Set the correct library path for OpenSSL
export LD_LIBRARY_PATH="$(brew --prefix openssl@3)/lib:$LD_LIBRARY_PATH"

# Activate virtual environment
source .venv/bin/activate

# Run the Python script
python "$@" 