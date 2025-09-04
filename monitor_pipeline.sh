#!/bin/bash

# Monitor pipeline logs and status
# Usage: ./monitor_pipeline.sh [latest|all|errors|status]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "🔍 Pipeline Monitor"
    echo "=================="
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  latest    Monitor latest pipeline log (default)"
    echo "  all       Monitor all pipeline logs"
    echo "  errors    Monitor error logs only"
    echo "  script    Monitor script logs only"
    echo "  status    Show pipeline status and PIDs"
    echo "  stop      Stop all running pipelines and child processes"
    echo "  help      Show this help message"
    echo
    echo "Examples:"
    echo "  $0              # Monitor latest log"
    echo "  $0 latest       # Monitor latest log"
    echo "  $0 errors       # Monitor error logs"
    echo "  $0 script       # Monitor script logs"
    echo "  $0 status       # Show running pipelines"
    echo "  $0 stop         # Stop all pipelines and child processes"
    echo
}

# Function to get latest log file
get_latest_log() {
    if [ ! -d "logs" ]; then
        print_error "No logs directory found"
        exit 1
    fi
    
    LATEST_LOG=$(ls -t logs/pipeline_*.log 2>/dev/null | head -1)
    if [ -z "$LATEST_LOG" ]; then
        print_error "No pipeline logs found"
        exit 1
    fi
    echo "$LATEST_LOG"
}

# Function to get latest error log
get_latest_error_log() {
    if [ ! -d "logs" ]; then
        print_error "No logs directory found"
        exit 1
    fi
    
    LATEST_ERROR_LOG=$(ls -t logs/pipeline_*_errors.log 2>/dev/null | head -1)
    if [ -z "$LATEST_ERROR_LOG" ]; then
        print_error "No error logs found"
        exit 1
    fi
    echo "$LATEST_ERROR_LOG"
}

# Function to get latest script log
get_latest_script_log() {
    if [ ! -d "logs" ]; then
        print_error "No logs directory found"
        exit 1
    fi
    
    LATEST_SCRIPT_LOG=$(ls -t logs/pipeline_*_script.log 2>/dev/null | head -1)
    if [ -z "$LATEST_SCRIPT_LOG" ]; then
        print_error "No script logs found"
        exit 1
    fi
    echo "$LATEST_SCRIPT_LOG"
}

# Function to show pipeline status
show_status() {
    print_status "Checking pipeline status..."
    
    if [ ! -d "logs" ]; then
        print_warning "No logs directory found - no pipelines have been run"
        return
    fi
    
    # Find all PID files
    PID_FILES=$(ls logs/pipeline_*.pid 2>/dev/null | sort -r)
    
    if [ -z "$PID_FILES" ]; then
        print_warning "No pipeline PID files found"
        return
    fi
    
    echo
    echo "📊 Pipeline Status:"
    echo "=================="
    
    for pid_file in $PID_FILES; do
        if [ -f "$pid_file" ]; then
            PID=$(cat "$pid_file")
            TIMESTAMP=$(basename "$pid_file" .pid | sed 's/pipeline_//')
            
            # Check if process is still running
            if ps -p "$PID" > /dev/null 2>&1; then
                STATUS="🟢 RUNNING"
                LOG_FILE="logs/pipeline_${TIMESTAMP}.log"
                ERROR_LOG="logs/pipeline_${TIMESTAMP}_errors.log"
                SCRIPT_LOG="logs/pipeline_${TIMESTAMP}_script.log"
                
                if [ -f "$LOG_FILE" ]; then
                    LAST_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null | cut -c1-80)
                    if [ -n "$LAST_LINE" ]; then
                        LAST_ACTIVITY="Last: $LAST_LINE"
                    else
                        LAST_ACTIVITY="No activity yet"
                    fi
                else
                    LAST_ACTIVITY="Log file not found"
                fi
            else
                STATUS="🔴 STOPPED"
                LAST_ACTIVITY="Process not running"
            fi
            
            echo "PID: $PID | $STATUS | Started: $TIMESTAMP"
            echo "   $LAST_ACTIVITY"
            
            # Show log file information
            if [ -f "logs/pipeline_${TIMESTAMP}.log" ]; then
                LOG_SIZE=$(du -h "logs/pipeline_${TIMESTAMP}.log" | cut -f1)
                echo "   📄 Main log: logs/pipeline_${TIMESTAMP}.log ($LOG_SIZE)"
            fi
            if [ -f "logs/pipeline_${TIMESTAMP}_errors.log" ]; then
                ERROR_SIZE=$(du -h "logs/pipeline_${TIMESTAMP}_errors.log" | cut -f1)
                echo "   ❌ Error log: logs/pipeline_${TIMESTAMP}_errors.log ($ERROR_SIZE)"
            fi
            if [ -f "logs/pipeline_${TIMESTAMP}_script.log" ]; then
                SCRIPT_SIZE=$(du -h "logs/pipeline_${TIMESTAMP}_script.log" | cut -f1)
                echo "   📝 Script log: logs/pipeline_${TIMESTAMP}_script.log ($SCRIPT_SIZE)"
            fi
            echo
        fi
    done
}

# Function to kill a process tree recursively
kill_process_tree() {
    local pid=$1
    local signal=${2:-TERM}
    
    # Get all child processes
    local children=$(pgrep -P "$pid" 2>/dev/null)
    
    # Kill children first
    for child in $children; do
        kill_process_tree "$child" "$signal"
    done
    
    # Kill the parent process
    if ps -p "$pid" > /dev/null 2>&1; then
        kill -"$signal" "$pid" 2>/dev/null
        sleep 0.1
        
        # If process still exists, force kill
        if ps -p "$pid" > /dev/null 2>&1; then
            kill -KILL "$pid" 2>/dev/null
        fi
    fi
}

# Function to stop all pipelines
stop_pipelines() {
    print_warning "Stopping all running pipelines and their child processes..."
    
    if [ ! -d "logs" ]; then
        print_warning "No logs directory found - no pipelines to stop"
        return
    fi
    
    PID_FILES=$(ls logs/pipeline_*.pid 2>/dev/null)
    
    if [ -z "$PID_FILES" ]; then
        print_warning "No pipeline PID files found"
        return
    fi
    
    STOPPED_COUNT=0
    for pid_file in $PID_FILES; do
        if [ -f "$pid_file" ]; then
            PID=$(cat "$pid_file")
            TIMESTAMP=$(basename "$pid_file" .pid | sed 's/pipeline_//')
            
            if ps -p "$PID" > /dev/null 2>&1; then
                print_status "Stopping pipeline PID $PID and all child processes (started: $TIMESTAMP)"
                
                # Kill the entire process tree
                kill_process_tree "$PID"
                
                # Wait a moment for processes to terminate
                sleep 0.5
                
                # Check if main process is still running
                if ps -p "$PID" > /dev/null 2>&1; then
                    print_warning "Pipeline PID $PID is still running, force killing..."
                    kill -KILL "$PID" 2>/dev/null
                fi
                
                STOPPED_COUNT=$((STOPPED_COUNT + 1))
                
                # Clean up PID file
                rm -f "$pid_file"
                print_success "Pipeline $TIMESTAMP stopped and PID file cleaned up"
            else
                print_warning "Pipeline PID $PID is not running, cleaning up PID file"
                rm -f "$pid_file"
            fi
        fi
    done
    
    # Additional cleanup: kill any remaining processes that might be related
    # Look for processes that might be spawned by the pipeline
    print_status "Checking for any remaining related processes..."
    
    # Kill any processes that might be running in the background
    # This is a more aggressive approach to ensure all services are stopped
    REMAINING_PIDS=$(ps aux | grep -E "(pipeline|monitor)" | grep -v grep | awk '{print $2}' 2>/dev/null)
    
    if [ -n "$REMAINING_PIDS" ]; then
        print_warning "Found remaining related processes, stopping them..."
        for remaining_pid in $REMAINING_PIDS; do
            if ps -p "$remaining_pid" > /dev/null 2>&1; then
                print_status "Stopping remaining process PID $remaining_pid"
                kill_process_tree "$remaining_pid"
            fi
        done
    fi
    
    if [ $STOPPED_COUNT -gt 0 ]; then
        print_success "Stopped $STOPPED_COUNT pipeline(s) and all child processes"
    else
        print_warning "No running pipelines found"
    fi
}

# Parse command line arguments
COMMAND=${1:-latest}

case $COMMAND in
    latest)
        LATEST_LOG=$(get_latest_log)
        print_status "Monitoring latest pipeline log: $LATEST_LOG"
        echo "Press Ctrl+C to stop monitoring"
        echo "=================================="
        tail -f -n +1 "$LATEST_LOG"
        ;;
    all)
        if [ ! -d "logs" ]; then
            print_error "No logs directory found"
            exit 1
        fi
        
        print_status "Monitoring all pipeline logs"
        echo "Press Ctrl+C to stop monitoring"
        echo "=================================="
        tail -f -n +1 logs/pipeline_*.log
        ;;
    errors)
        LATEST_ERROR_LOG=$(get_latest_error_log)
        print_status "Monitoring error log: $LATEST_ERROR_LOG"
        echo "Press Ctrl+C to stop monitoring"
        echo "=================================="
        tail -f -n +1 "$LATEST_ERROR_LOG"
        ;;
    script)
        LATEST_SCRIPT_LOG=$(get_latest_script_log)
        print_status "Monitoring script log: $LATEST_SCRIPT_LOG"
        echo "Press Ctrl+C to stop monitoring"
        echo "=================================="
        tail -f -n +1 "$LATEST_SCRIPT_LOG"
        ;;
    status)
        show_status
        ;;
    stop)
        stop_pipelines
        ;;
    help)
        show_usage
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac 