#!/bin/bash

# Check if input file and config are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_root_file> <config_yaml>"
    exit 1
fi

INPUT_FILE=$1
CONFIG_FILE=$2

# Train the model
echo "Training model with input file: $INPUT_FILE and config: $CONFIG_FILE"
python trainer.py --input_file "$INPUT_FILE" --config "$CONFIG_FILE"

# Extract values from the config.yaml
BATCH_SIZE=$(grep -E '^batch_size:' "$CONFIG_FILE" | awk '{print $2}')
LEARNING_RATE=$(grep -E '^learning_rate:' "$CONFIG_FILE" | awk '{print $2}')
NUM_EPOCHS=$(grep -E '^num_epochs:' "$CONFIG_FILE" | awk '{print $2}')
DATE_STR=$(date +%Y%m%d)

# Construct the expected output filenames using config values
LOSS_CSV="loss_batchSize${BATCH_SIZE}_learningRate${LEARNING_RATE}_epochs${NUM_EPOCHS}_${DATE_STR}.csv"
PRED_CSV="predictions_batchSize${BATCH_SIZE}_learningRate${LEARNING_RATE}_epochs${NUM_EPOCHS}_${DATE_STR}.csv"

# Check if the CSV files exist
if [ ! -f "$LOSS_CSV" ] || [ ! -f "$PRED_CSV" ]; then
    echo "Error: Expected CSV files not found:"
    echo "  - Expected Loss CSV: $LOSS_CSV"
    echo "  - Expected Predictions CSV: $PRED_CSV"
    exit 1
fi

echo "Using Loss CSV: $LOSS_CSV"
echo "Using Predictions CSV: $PRED_CSV"

# Generate the loss plot
echo "Generating loss plot..."
python loss_plotter.py "$LOSS_CSV"

# Generate the predictions plot
echo "Generating predictions plot..."
python plotter.py "$PRED_CSV"

echo "Plots generated successfully."
