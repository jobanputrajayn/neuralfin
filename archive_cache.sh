#!/bin/bash

# Archive Cache Script for Hyperparameter Tuning Project
# Archives all caches, logs, and checkpoints that would be cleared by clear_cache.sh

set -e

BACKUP_DIR="backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TARBALL="$BACKUP_DIR/cache_backup_$TIMESTAMP.tar.gz"

# List of files/directories to archive (same as clear_cache.sh)
ARCHIVE_ITEMS=(
  "data_cache"
  "news_cache"
  "hyperparameter_tuning_results"
  "extended_training_results"
  "backtesting_results"
  "analysis_plots"
  "final_model"
  "best_config.json"
  "hyperparameter_tuning.log"
  "__pycache__"
  "tensorboard_logs"
)

# Find .pyc, .log, .tmp, .temp, .png files (recursively)
EXTRA_ITEMS=(
  $(find . -name "*.pyc" -type f 2>/dev/null)
  $(find . -name "*.log" -type f 2>/dev/null)
  $(find . -name "*.tmp" -type f 2>/dev/null)
  $(find . -name "*.temp" -type f 2>/dev/null)
  $(find . -name "*.png" -type f 2>/dev/null)
)

# Create backups directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Build list of existing items
EXISTING_ITEMS=()
for item in "${ARCHIVE_ITEMS[@]}"; do
  if [ -e "$item" ]; then
    EXISTING_ITEMS+=("$item")
  fi
done
for item in "${EXTRA_ITEMS[@]}"; do
  if [ -e "$item" ]; then
    EXISTING_ITEMS+=("$item")
  fi
done

if [ ${#EXISTING_ITEMS[@]} -eq 0 ]; then
  echo "No cache, logs, or checkpoints found to archive."
  exit 0
fi

tar -czf "$TARBALL" "${EXISTING_ITEMS[@]}"

# Print summary
echo "🗄️  Archived the following items into $TARBALL:"
for item in "${EXISTING_ITEMS[@]}"; do
  echo "  - $item"
done

echo "�� Archive complete!" 