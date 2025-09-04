#!/bin/bash
#
# Start TensorBoard to monitor hyperparameter tuning runs.
#
# This script starts TensorBoard, pointing it to the log directory where
# the training and validation metrics are stored.

# --- Configuration ---
LOG_DIR="./tensorboard_logs"
PORT=6009
HOST="0.0.0.0"

# --- Main Logic ---
echo "🚀 Starting TensorBoard..."
echo "========================================="
echo "📁 Log Directory:  ${LOG_DIR}"
echo "🌐 Port:           ${PORT}"
echo "💻 Host:           ${HOST}"
echo "========================================="
echo ""
echo "📈 Access TensorBoard in your browser at: http://localhost:${PORT}"
echo "   (or http://<your-server-ip>:${PORT} if running remotely)"
echo ""
echo "Press Ctrl+C to stop TensorBoard."
echo ""

# Create log directory if it doesn't exist to prevent an error
mkdir -p "${LOG_DIR}"

# Start TensorBoard
tensorboard --logdir "${LOG_DIR}" --port "${PORT}" --bind_all --load_fast=false