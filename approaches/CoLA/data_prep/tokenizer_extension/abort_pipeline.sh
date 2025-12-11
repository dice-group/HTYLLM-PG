#!/bin/bash

echo "Cancelling tokenizer-extension jobs..."
squeue -u $USER -h -o "%i %j" | awk '/tokext/ {print $1}' | xargs scancel
echo "Done."
