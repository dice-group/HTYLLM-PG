#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Run the build script
# This initializes a downsized Mixtral MoE model
python src/build.py

echo "Model build complete!"
