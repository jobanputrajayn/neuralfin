#!/bin/bash

# Run complete pipeline in background with proper logging
# This script runs the full hyperparameter tuning pipeline and logs everything

# Create logs directory if it doesn't exist
mkdir -p logs

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/pipeline_${TIMESTAMP}.log"
ERROR_LOG="logs/pipeline_${TIMESTAMP}_errors.log"
SCRIPT_LOG="logs/pipeline_${TIMESTAMP}_script.log"

echo "Starting complete pipeline in background..."
echo "Log file: $LOG_FILE"
echo "Error log: $ERROR_LOG"
echo "Script log: $SCRIPT_LOG"
echo "Process ID will be saved to: logs/pipeline_${TIMESTAMP}.pid"

# Run the complete pipeline with no confirmation and redirect all output
# Use script command to capture rich text output properly
# Separate stdout and stderr logging - redirect from within the script command
nohup script -q -c "bash run_complete_pipeline.sh --start-step 1 --no-confirm > >(tee '$LOG_FILE') 2> >(tee '$ERROR_LOG' >&2)" "$SCRIPT_LOG" &
PIPELINE_PID=$!

# Save the PID for later reference
echo $PIPELINE_PID > "logs/pipeline_${TIMESTAMP}.pid"

# Disown the process so it continues running even if this shell is closed
disown $PIPELINE_PID

echo "Pipeline started with PID: $PIPELINE_PID"
echo "To monitor progress, use: ./monitor_pipeline.sh"
echo "To monitor errors, use: ./monitor_pipeline.sh errors"
echo "To monitor script output, use: ./monitor_pipeline.sh script"
echo "To check pipeline status, use: ./monitor_pipeline.sh status"
echo "To stop the pipeline, use: ./monitor_pipeline.sh stop" 