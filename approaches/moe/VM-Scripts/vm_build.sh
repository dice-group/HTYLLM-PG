#!/bin/bash

# Change to project root directory
cd "$(dirname "$0")/.."

# Run the build script
# This initializes a downsized Mixtral MoE model
python src/build.py

echo "Model build complete!"
