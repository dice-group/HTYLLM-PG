#!/bin/bash

export N_PROCS=20000 # Number of total processes to spawn. This is just an example, your utility should be ready to scale up or down when this value is changed.

# Loading the environment
source environment_initialization.sh

# Call to some preparation logic (optional)
bash prepare.sh

# Spawner calling your preprocessing script N_PROCS times across the computing cluster
super_duper_process_spawner -n $N_PROCS python text_preprocessor.py

echo "Finished preprocessing data."