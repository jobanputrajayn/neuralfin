#!/bin/bash

# Complete Hyperparameter Tuning Pipeline
# Uses subset parameters for testing (from launch.json)

set -e  # Exit on any error

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

print_header() {
    echo -e "${PURPLE}[HEADER]${NC} $1"
}

print_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "🚀 Complete Hyperparameter Tuning Pipeline"
    echo "=========================================="
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --start-step STEP    Start from specific step (1-6)"
    echo "  --no-confirm         Run without user confirmation"
    echo "  --help               Show this help message"
    echo
    echo "Steps:"
    echo "  1. Hyperparameter Tuning (subset parameters)"
    echo "  2. Extended Training"
    echo "  3. Results Analysis"
    echo "  4. Final Model Training"
    echo "  5. Backtesting"
    echo "  6. Signal Generation"
    echo
    echo "Examples:"
    echo "  $0                    # Run full pipeline with confirmation"
    echo "  $0 --start-step 3     # Start from Results Analysis"
    echo "  $0 --start-step 4 --no-confirm  # Start from Final Training, no confirmation"
    echo "  $0 --help             # Show this help"
    echo
}

# Parse command line arguments
START_STEP=1
NO_CONFIRM=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --start-step)
            START_STEP="$2"
            shift 2
            ;;
        --no-confirm)
            NO_CONFIRM=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate start step
if ! [[ "$START_STEP" =~ ^[1-6]$ ]]; then
    print_error "Invalid start step: $START_STEP. Must be 1-6."
    exit 1
fi

# Function to check if previous step succeeded
check_previous_step() {
    if [ $? -ne 0 ]; then
        print_error "Previous step failed. Exiting pipeline."
        exit 1
    fi
}

# Function to wait for user confirmation
wait_for_confirmation() {
    if [ "$NO_CONFIRM" = true ]; then
        return
    fi
    
    echo
    read -p "Press Enter to continue to the next step, or 'q' to quit: " -r
    if [[ $REPLY =~ ^[Qq]$ ]]; then
        print_warning "Pipeline stopped by user."
        exit 0
    fi
}

# Function to check if step prerequisites are met
check_step_prerequisites() {
    local step=$1
    
    case $step in
        1)
            # Step 1 has no prerequisites
            return 0
            ;;
        2)
            # Step 2 requires hyperparameter tuning results
            if [ ! -f "hyperparameter_tuning_results/trial_results.json" ]; then
                print_error "Step 2 requires hyperparameter tuning results. Please run step 1 first."
                return 1
            fi
            ;;
        3)
            # Step 3 requires both hyperparameter and extended training results
            if [ ! -f "hyperparameter_tuning_results/trial_results.json" ]; then
                print_error "Step 3 requires hyperparameter tuning results. Please run step 1 first."
                return 1
            fi
            if [ ! -f "extended_training_results/extended_training_results.json" ]; then
                print_error "Step 3 requires extended training results. Please run step 2 first."
                return 1
            fi
            ;;
        4)
            # Step 4 requires best configuration file
            if [ ! -f "best_config.json" ]; then
                print_error "Step 4 requires best configuration file. Please run step 3 first."
                return 1
            fi
            ;;
        5)
            # Step 5 requires final model
            if [ ! -f "final_model/config.json" ]; then
                print_error "Step 5 requires final model. Please run step 4 first."
                return 1
            fi
            ;;
        6)
            # Step 6 requires final model
            if [ ! -f "final_model/config.json" ]; then
                print_error "Step 6 requires final model. Please run step 4 first."
                return 1
            fi
            ;;
    esac
    return 0
}

# Function to run step 1: Hyperparameter Tuning
run_step_1() {
    print_step "1: Starting Hyperparameter Tuning"
    print_status "Using subset parameters for testing:"
    echo "   - Ticker count: -1"
    echo "   - Random trials: 20"
    echo "   - Bayesian trials: 30"
    echo "   - Fine-tune trials: 30"
    echo "   - Epochs per trial: 200"
    echo "   - Data period: 10y"

    python run_hyperparameter_tuning.py \
        --ticker-count -1 \
        --random-trials 30 \
        --bayesian-trials 20 \
        --fine-tune-trials 5 \
        --epochs-random 5 \
        --epochs-bayesian 5 \
        --epochs-fine-tune 20 \
        --data-period 10y \
        --prefetch-buffer-size 1000

    check_previous_step
    print_success "Hyperparameter tuning completed"

    # Check if results were created
    if [ ! -f "hyperparameter_tuning_results/trial_results.json" ]; then
        print_error "Hyperparameter tuning results not found"
        exit 1
    fi
}

# Function to run step 2: Extended Training
run_step_2() {
    print_step "2: Starting Extended Training"
    print_status "Training top 3 configurations with cross-validation"

    python run_extended_training.py \
        --top-n 5 \
        --epochs 20 \
        --results-dir ./hyperparameter_tuning_results

    check_previous_step
    print_success "Extended training completed"

    # Check if extended results were created
    if [ ! -f "extended_training_results/extended_training_results.json" ]; then
        print_error "Extended training results not found"
        exit 1
    fi
}

# Function to run step 3: Results Analysis
run_step_3() {
    print_step "3: Analyzing Results"
    print_status "Comparing hyperparameter tuning vs extended training results"

    python analyze_results.py \
        --hyperparameter-dir ./hyperparameter_tuning_results \
        --extended-dir ./extended_training_results \
        --output-config best_config.json \
        --plots-dir ./analysis_plots

    check_previous_step
    print_success "Results analysis completed"

    # Check if best config was created
    if [ ! -f "best_config.json" ]; then
        print_error "Best configuration file not found"
        exit 1
    fi
}

# Function to run step 4: Final Model Training
run_step_4() {
    print_step "4: Training Final Model"
    print_status "Training production model with best configuration"

    python run_final_training.py \
        --config best_config.json \
        --output-dir ./final_model \
        --epochs 40 \
        --patience 10

    check_previous_step
    print_success "Final model training completed"

    # Check if final model was created
    if [ ! -f "final_model/config.json" ]; then
        print_error "Final model not found"
        exit 1
    fi
}

# Function to run step 5: Backtesting
run_step_5() {
    print_step "5: Running Backtesting"
    print_status "Evaluating model performance on historical data"

    # Extract configuration from best_config.json for consistency
    if [ -f "best_config.json" ]; then
        # Extract tickers and sequence length from best_config.json
        MODEL_CONFIG=$(python -c "
import json
config = json.load(open('best_config.json'))
tickers = ' '.join(config.get('tickers', ['MSFT', 'NVDA', 'AAPL', 'AMZN', 'META']))
seq_length = config.get('seq_length', 60)
print(f'{tickers}|{seq_length}')
")
        
        # Parse the result: tickers|seq_length
        TICKERS=$(echo "$MODEL_CONFIG" | cut -d'|' -f1)
        SEQ_LENGTH=$(echo "$MODEL_CONFIG" | cut -d'|' -f2)
        
        print_status "Using configuration from trained model:"
        echo "   - Tickers: $TICKERS"
        echo "   - Sequence length: $SEQ_LENGTH days"
    else
        # Fallback to step 1 defaults
        TICKERS="MSFT NVDA AAPL AMZN META"
        SEQ_LENGTH=60
        print_warning "Using fallback configuration (step 1 defaults)"
    fi

    # Compute appropriate periods based on actual sequence length
    MIN_REQUIRED_DAYS=$((SEQ_LENGTH * 6))  # 3x sequence length for meaningful backtesting
    
    if [ $MIN_REQUIRED_DAYS -ge 365 ]; then
        BACKTEST_PERIOD="1y"
    elif [ $MIN_REQUIRED_DAYS -ge 180 ]; then
        BACKTEST_PERIOD="6mo"
    elif [ $MIN_REQUIRED_DAYS -ge 90 ]; then
        BACKTEST_PERIOD="3mo"
    elif [ $MIN_REQUIRED_DAYS -ge 60 ]; then
        BACKTEST_PERIOD="2mo"
    else
        BACKTEST_PERIOD="1mo"
    fi
    
    print_status "Computed backtest period based on model sequence length:"
    echo "   - Model sequence length: $SEQ_LENGTH days"
    echo "   - Backtest period: $BACKTEST_PERIOD (requires $MIN_REQUIRED_DAYS days)"

    python run_backtesting.py \
        --data-period "$BACKTEST_PERIOD" \
        --tickers $TICKERS \
        --initial-cash 10000 \
        --commission-rate 0.001 \
        --output-dir ./backtesting_results

    check_previous_step
    print_success "Backtesting completed"
}

# Function to run step 6: Signal Generation
run_step_6() {
    print_step "6: Generating Trading Signals"
    print_status "Generating latest trading signals for the trained model"

    # Extract configuration from best_config.json for consistency
    if [ -f "best_config.json" ]; then
        # Extract tickers and sequence length from best_config.json
        MODEL_CONFIG=$(python -c "
import json
config = json.load(open('best_config.json'))
tickers = ' '.join(config.get('tickers', ['MSFT', 'NVDA', 'AAPL', 'AMZN', 'META']))
seq_length = config.get('seq_length', 60)
print(f'{tickers}|{seq_length}')
")
        
        # Parse the result: tickers|seq_length
        TICKERS=$(echo "$MODEL_CONFIG" | cut -d'|' -f1)
        SEQ_LENGTH=$(echo "$MODEL_CONFIG" | cut -d'|' -f2)
        
        print_status "Using configuration from trained model:"
        echo "   - Tickers: $TICKERS"
        echo "   - Sequence length: $SEQ_LENGTH days"
    else
        # Fallback to step 1 defaults
        TICKERS="MSFT NVDA AAPL AMZN META"
        SEQ_LENGTH=60
        print_warning "Using fallback configuration (step 1 defaults)"
    fi

    # Compute signal generation period (sequence length + buffer)
    SIGNAL_REQUIRED_DAYS=$((SEQ_LENGTH + 30))
    
    if [ $SIGNAL_REQUIRED_DAYS -ge 365 ]; then
        SIGNAL_PERIOD="1y"
    elif [ $SIGNAL_REQUIRED_DAYS -ge 180 ]; then
        SIGNAL_PERIOD="6mo"
    elif [ $SIGNAL_REQUIRED_DAYS -ge 90 ]; then
        SIGNAL_PERIOD="3mo"
    elif [ $SIGNAL_REQUIRED_DAYS -ge 60 ]; then
        SIGNAL_PERIOD="2mo"
    else
        SIGNAL_PERIOD="1mo"
    fi
    
    print_status "Computed signal period based on model sequence length:"
    echo "   - Model sequence length: $SEQ_LENGTH days"
    echo "   - Signal period: $SIGNAL_PERIOD (requires $SIGNAL_REQUIRED_DAYS days)"

    python run_backtesting.py \
        --generate-signals \
        --tickers $TICKERS \
        --signals-period $SIGNAL_PERIOD \
        --output-dir ./backtesting_results

    check_previous_step
    print_success "Signal generation completed"
}

# Check if GPU is available and set CUDA library path only if needed
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "🚀 GPU detected - Setting up CUDA library path"
    
    # Try to detect CUDA installation path dynamically
    CUDA_LIB_PATH=""
    
    # Method 1: Check common CUDA installation paths
    for cuda_path in "/usr/local/cuda" "/opt/cuda" "/usr/cuda"; do
        if [ -d "$cuda_path/targets/x86_64-linux/lib" ]; then
            CUDA_LIB_PATH="$cuda_path/targets/x86_64-linux/lib"
            echo "📁 Found CUDA at: $cuda_path"
            break
        fi
    done
    
    # Method 2: Use nvcc to find CUDA installation
    if [ -z "$CUDA_LIB_PATH" ] && command -v nvcc &> /dev/null; then
        nvcc_path=$(which nvcc)
        if [ -n "$nvcc_path" ]; then
            # nvcc is typically in bin/, so go up to lib/
            cuda_root=$(dirname $(dirname "$nvcc_path"))
            if [ -d "$cuda_root/targets/x86_64-linux/lib" ]; then
                CUDA_LIB_PATH="$cuda_root/targets/x86_64-linux/lib"
                echo "📁 Found CUDA via nvcc at: $cuda_root"
            fi
        fi
    fi
    
    # Method 3: Check environment variables
    if [ -z "$CUDA_LIB_PATH" ] && [ -n "$CUDA_HOME" ]; then
        if [ -d "$CUDA_HOME/targets/x86_64-linux/lib" ]; then
            CUDA_LIB_PATH="$CUDA_HOME/targets/x86_64-linux/lib"
            echo "📁 Found CUDA via CUDA_HOME at: $CUDA_HOME"
        fi
    fi
    
    # Method 4: Check CUDA_PATH
    if [ -z "$CUDA_LIB_PATH" ] && [ -n "$CUDA_PATH" ]; then
        if [ -d "$CUDA_PATH/targets/x86_64-linux/lib" ]; then
            CUDA_LIB_PATH="$CUDA_PATH/targets/x86_64-linux/lib"
            echo "📁 Found CUDA via CUDA_PATH at: $CUDA_PATH"
        fi
    fi
    
    # Set LD_LIBRARY_PATH if CUDA library path was found
    if [ -n "$CUDA_LIB_PATH" ]; then
        export LD_LIBRARY_PATH="$CUDA_LIB_PATH:$LD_LIBRARY_PATH"
        echo "✅ CUDA library path set to: $CUDA_LIB_PATH"
    else
        echo "⚠️  CUDA installation not found - GPU may not work properly"
        echo "   Tried paths: /usr/local/cuda, /opt/cuda, /usr/cuda, and environment variables"
    fi
else
    echo "ℹ️  No GPU detected - Running on CPU/TPU"
fi

print_header "Starting Complete Hyperparameter Tuning Pipeline"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "run_hyperparameter_tuning.py" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

# Check if required directories exist, create if needed
mkdir -p hyperparameter_tuning_results
mkdir -p extended_training_results
mkdir -p backtesting_results
mkdir -p analysis_plots
mkdir -p final_model

print_status "Environment check completed"

echo
echo "📋 Pipeline Steps:"
echo "1. Hyperparameter Tuning (subset parameters)"
echo "2. Extended Training"
echo "3. Results Analysis"
echo "4. Final Model Training"
echo "5. Backtesting"
echo "6. Signal Generation"
echo

print_status "Starting from step: $START_STEP"
if [ "$NO_CONFIRM" = true ]; then
    print_warning "Running without user confirmation"
fi
echo

# Check prerequisites for starting step
if ! check_step_prerequisites $START_STEP; then
    exit 1
fi

# Run steps based on start step
case $START_STEP in
    1)
        run_step_1
        wait_for_confirmation
        run_step_2
        wait_for_confirmation
        run_step_3
        wait_for_confirmation
        run_step_4
        wait_for_confirmation
        run_step_5
        wait_for_confirmation
        run_step_6
        ;;
    2)
        run_step_2
        wait_for_confirmation
        run_step_3
        wait_for_confirmation
        run_step_4
        wait_for_confirmation
        run_step_5
        wait_for_confirmation
        run_step_6
        ;;
    3)
        run_step_3
        wait_for_confirmation
        run_step_4
        wait_for_confirmation
        run_step_5
        wait_for_confirmation
        run_step_6
        ;;
    4)
        run_step_4
        wait_for_confirmation
        run_step_5
        wait_for_confirmation
        run_step_6
        ;;
    5)
        run_step_5
        wait_for_confirmation
        run_step_6
        ;;
    6)
        run_step_6
        ;;
esac

# Final summary
echo
echo "=================================================="
print_success "🎉 Pipeline Successfully Completed!"
echo "=================================================="

echo
echo "📁 Generated Files:"
echo "   - Hyperparameter tuning: ./hyperparameter_tuning_results/"
echo "   - Best configuration: ./best_config.json"
echo "   - Analysis plots: ./analysis_plots/"
echo "   - Final model: ./final_model/"
echo "   - Backtesting results: ./backtesting_results/"
echo "   - Trading signals: ./backtesting_results/latest_signals.csv"

echo
echo "🚀 Next Steps:"
echo "   - Review backtesting results in ./backtesting_results/"
echo "   - Check latest signals in ./backtesting_results/latest_signals.csv"
echo "   - Analyze performance plots in ./analysis_plots/"
echo "   - For production, update parameters in this script and re-run"
echo "   - Run step 6 independently for fresh signals: $0 --start-step 6"

echo
print_success "Pipeline execution completed successfully!" 