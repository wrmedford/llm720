#!/bin/bash

# Script for running foundation model training with PEER and MLA
# This script handles distributed training setup and error handling

set -e  # Exit on error

# Default configuration file
CONFIG_FILE="config.yaml"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --config)
      CONFIG_FILE="$2"
      shift
      shift
      ;;
    --nodes)
      NUM_NODES="$2"
      shift
      shift
      ;;
    --gpus-per-node)
      GPUS_PER_NODE="$2"
      shift
      shift
      ;;
    --master-addr)
      MASTER_ADDR="$2"
      shift
      shift
      ;;
    --master-port)
      MASTER_PORT="$2"
      shift
      shift
      ;;
    --node-rank)
      NODE_RANK="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set default values if not provided
NUM_NODES=${NUM_NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
NODE_RANK=${NODE_RANK:-0}

# Calculate world size
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

# Check if configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Configuration file '$CONFIG_FILE' not found."
  exit 1
fi

# Create log directory
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Set up environment variables for distributed training
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$WORLD_SIZE
export NODE_RANK=$NODE_RANK

# Function to handle training failures
handle_failure() {
  echo "Training process failed with exit code $1"
  echo "Check logs in $LOG_DIR for details"
  
  # Archive logs and failed checkpoint for debugging
  FAILURE_DIR="failures/$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$FAILURE_DIR"
  cp -r "$LOG_DIR" "$FAILURE_DIR/"
  
  # Find the latest checkpoint and copy it to the failure directory
  LATEST_CHECKPOINT=$(find output -name "checkpoint-*" -type d | sort -V | tail -n 1)
  if [ -n "$LATEST_CHECKPOINT" ]; then
    cp -r "$LATEST_CHECKPOINT" "$FAILURE_DIR/"
  fi
  
  echo "Failure information saved to $FAILURE_DIR"
  exit 1
}

# Launch distributed training
echo "Starting training with $NUM_NODES nodes, $GPUS_PER_NODE GPUs per node, $WORLD_SIZE total processes"
echo "Using configuration file: $CONFIG_FILE"
echo "Logs will be saved to: $LOG_DIR"

# Use torchrun for distributed training
torchrun \
  --nnodes=$NUM_NODES \
  --nproc_per_node=$GPUS_PER_NODE \
  --rdzv_id=job_$(date +%Y%m%d_%H%M%S) \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --node_rank=$NODE_RANK \
  scripts/train.py \
  --config "$CONFIG_FILE" \
  2>&1 | tee "$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log" || handle_failure $?

echo "Training completed successfully!"