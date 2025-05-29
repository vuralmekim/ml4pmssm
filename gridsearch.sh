#!/bin/bash

# Usage: ./gridsearch.sh input.root base_config.yaml

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_root_file> <base_config_yaml>"
    exit 1
fi

INPUT_FILE=$1
BASE_CONFIG=$2
CONFIG_TEMPLATE="tmp_config.yaml"

LEARNING_RATES=(1e-5 1e-4 1e-3)
BATCH_SIZES=(512 1024 2048)
NUM_EPOCHS=100  # quick training

for LR in "${LEARNING_RATES[@]}"; do
  for BS in "${BATCH_SIZES[@]}"; do
    echo "Running training with learning_rate=$LR, batch_size=$BS"

    # Generate new config
    cat > "$CONFIG_TEMPLATE" <<EOF
learning_rate: $LR
weight_decay: 0.01
scheduler_T0: 10
num_epochs: $NUM_EPOCHS
batch_size: $BS
EOF

    # Run the train-and-plot script
    ./train_and_plot.sh "$INPUT_FILE" "$CONFIG_TEMPLATE"
  done
done

