#!/usr/bin/env python3
"""
Complete Hyperparameter Tuning Pipeline - Python Version
Equivalent to run_complete_pipeline_subset.sh but debuggable in VS Code
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Optional

# Colors for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

def print_status(message: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")

def print_success(message: str):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")

def print_warning(message: str):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")

def print_error(message: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")

def print_header(message: str):
    print(f"{Colors.PURPLE}[HEADER]{Colors.NC} {message}")

def print_step(message: str):
    print(f"{Colors.CYAN}[STEP]{Colors.NC} {message}")

def show_usage():
    """Show usage information."""
    print("🚀 Complete Hyperparameter Tuning Pipeline")
    print("==========================================")
    print()
    print("Usage: python run_complete_pipeline.py [OPTIONS]")
    print()
    print("Options:")
    print("  --start-step STEP    Start from specific step (1-6)")
    print("  --no-confirm         Run without user confirmation")
    print("  --help               Show this help message")
    print()
    print("Steps:")
    print("  1. Hyperparameter Tuning (subset parameters)")
    print("  2. Extended Training")
    print("  3. Results Analysis")
    print("  4. Final Model Training")
    print("  5. Backtesting")
    print("  6. Signal Generation")
    print()
    print("Examples:")
    print("  python run_complete_pipeline.py                    # Run full pipeline with confirmation")
    print("  python run_complete_pipeline.py --start-step 3     # Start from Results Analysis")
    print("  python run_complete_pipeline.py --start-step 4 --no-confirm  # Start from Final Training, no confirmation")
    print("  python run_complete_pipeline.py --help             # Show this help")
    print()

def check_previous_step(return_code: int) -> None:
    """Check if previous step succeeded."""
    if return_code != 0:
        print_error("Previous step failed. Exiting pipeline.")
        sys.exit(1)

def wait_for_confirmation(no_confirm: bool) -> None:
    """Wait for user confirmation unless no_confirm is True."""
    if no_confirm:
        return
    
    print()
    try:
        reply = input("Press Enter to continue to the next step, or 'q' to quit: ")
        if reply.lower() == 'q':
            print_warning("Pipeline stopped by user.")
            sys.exit(0)
    except KeyboardInterrupt:
        print_warning("Pipeline stopped by user.")
        sys.exit(0)

def check_step_prerequisites(step: int) -> bool:
    """Check if step prerequisites are met."""
    if step == 1:
        # Step 1 has no prerequisites
        return True
    elif step == 2:
        # Step 2 requires hyperparameter tuning results
        if not Path("hyperparameter_tuning_results/trial_results.json").exists():
            print_error("Step 2 requires hyperparameter tuning results. Please run step 1 first.")
            return False
    elif step == 3:
        # Step 3 requires both hyperparameter and extended training results
        if not Path("hyperparameter_tuning_results/trial_results.json").exists():
            print_error("Step 3 requires hyperparameter tuning results. Please run step 1 first.")
            return False
        if not Path("extended_training_results/extended_training_results.json").exists():
            print_error("Step 3 requires extended training results. Please run step 2 first.")
            return False
    elif step == 4:
        # Step 4 requires best configuration file
        if not Path("best_config.json").exists():
            print_error("Step 4 requires best configuration file. Please run step 3 first.")
            return False
    elif step == 5:
        # Step 5 requires final model
        if not Path("final_model/config.json").exists():
            print_error("Step 5 requires final model. Please run step 4 first.")
            return False
    elif step == 6:
        # Step 6 requires final model
        if not Path("final_model/config.json").exists():
            print_error("Step 6 requires final model. Please run step 4 first.")
            return False
    
    return True

def run_python_script(script_name: str, args: List[str]) -> int:
    """Run a Python script by importing it directly (no subprocess)."""
    print(f"🔧 Running: {script_name} {' '.join(args)}")
    
    # Parse arguments into a format that can be passed to the module
    import argparse
    import importlib.util
    
    try:
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("script_module", script_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Set up sys.argv to match what the script expects
        original_argv = sys.argv.copy()
        sys.argv = [script_name] + args
        
        # Run the main function if it exists
        if hasattr(module, 'main'):
            module.main()
            return 0
        else:
            print_error(f"No main() function found in {script_name}")
            return 1
            
    except Exception as e:
        print_error(f"Error running {script_name}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

def run_step_1() -> int:
    """Run step 1: Hyperparameter Tuning."""
    print_step("1: Starting Hyperparameter Tuning")
    print_status("Using subset parameters for testing:")
    print("   - Ticker count: 2")
    print("   - Random trials: 1")
    print("   - Bayesian trials: 1")
    print("   - Fine-tune trials: 1")
    print("   - Epochs per trial: 2")
    print("   - Data period: 1y")

    args = [
        "--ticker-count", "2",
        "--random-trials", "1",
        "--bayesian-trials", "1",
        "--fine-tune-trials", "1",
        "--epochs-random", "2",
        "--epochs-bayesian", "2",
        "--epochs-fine-tune", "2",
        "--data-period", "1y",
        "--prefetch-buffer-size", "100"
    ]
    
    return_code = run_python_script("run_hyperparameter_tuning.py", args)
    check_previous_step(return_code)
    print_success("Hyperparameter tuning completed")

    # Check if results were created
    if not Path("hyperparameter_tuning_results/trial_results.json").exists():
        print_error("Hyperparameter tuning results not found")
        return 1
    
    return return_code

def run_step_2() -> int:
    """Run step 2: Extended Training."""
    print_step("2: Starting Extended Training")
    print_status("Training top 3 configurations with cross-validation")

    args = [
        "--top-n", "3",
        "--epochs", "2",
        "--results-dir", "./hyperparameter_tuning_results"
    ]
    
    return_code = run_python_script("run_extended_training.py", args)
    check_previous_step(return_code)
    print_success("Extended training completed")

    # Check if extended results were created
    if not Path("extended_training_results/extended_training_results.json").exists():
        print_error("Extended training results not found")
        return 1
    
    return return_code

def run_step_3() -> int:
    """Run step 3: Results Analysis."""
    print_step("3: Analyzing Results")
    print_status("Comparing hyperparameter tuning vs extended training results")

    args = [
        "--hyperparameter-dir", "./hyperparameter_tuning_results",
        "--extended-dir", "./extended_training_results",
        "--output-config", "best_config.json",
        "--plots-dir", "./analysis_plots"
    ]
    
    return_code = run_python_script("analyze_results.py", args)
    check_previous_step(return_code)
    print_success("Results analysis completed")

    # Check if best config was created
    if not Path("best_config.json").exists():
        print_error("Best configuration file not found")
        return 1
    
    return return_code

def run_step_4() -> int:
    """Run step 4: Final Model Training."""
    print_step("4: Training Final Model")
    print_status("Training production model with best configuration")

    args = [
        "--config", "best_config.json",
        "--output-dir", "./final_model",
        "--epochs", "3",
        "--patience", "1"
    ]
    
    return_code = run_python_script("run_final_training.py", args)
    check_previous_step(return_code)
    print_success("Final model training completed")

    # Check if final model was created
    if not Path("final_model/config.json").exists():
        print_error("Final model not found")
        return 1
    
    return return_code

def run_step_5() -> int:
    """Run step 5: Backtesting."""
    print_step("5: Running Backtesting")
    print_status("Evaluating model performance on historical data")

    # Extract configuration from best_config.json for consistency
    tickers = "MSFT NVDA AAPL AMZN META"
    seq_length = 60
    
    if Path("best_config.json").exists():
        try:
            with open("best_config.json", "r") as f:
                config = json.load(f)
            
            tickers = " ".join(config.get("tickers", ["MSFT", "NVDA", "AAPL", "AMZN", "META"]))
            seq_length = config.get("seq_length", 60)
            
            print_status("Using configuration from trained model:")
            print(f"   - Tickers: {tickers}")
            print(f"   - Sequence length: {seq_length} days")
        except Exception as e:
            print_warning(f"Could not read best_config.json: {e}")
            print_warning("Using fallback configuration (step 1 defaults)")
    else:
        print_warning("Using fallback configuration (step 1 defaults)")

    # Compute appropriate periods based on actual sequence length
    min_required_days = seq_length * 6  # 3x sequence length for meaningful backtesting
    
    if min_required_days >= 365:
        backtest_period = "1y"
    elif min_required_days >= 180:
        backtest_period = "6mo"
    elif min_required_days >= 90:
        backtest_period = "3mo"
    elif min_required_days >= 60:
        backtest_period = "2mo"
    else:
        backtest_period = "1mo"
    
    print_status("Computed backtest period based on model sequence length:")
    print(f"   - Model sequence length: {seq_length} days")
    print(f"   - Backtest period: {backtest_period} (requires {min_required_days} days)")

    args = [
        "--data-period", backtest_period,
        "--tickers"] + tickers.split() + [
        "--initial-cash", "10000",
        "--commission-rate", "0.001",
        "--output-dir", "./backtesting_results"
    ]
    
    return_code = run_python_script("run_backtesting.py", args)
    check_previous_step(return_code)
    print_success("Backtesting completed")
    
    return return_code

def run_step_6() -> int:
    """Run step 6: Signal Generation."""
    print_step("6: Generating Trading Signals")
    print_status("Generating latest trading signals for the trained model")

    # Extract configuration from best_config.json for consistency
    tickers = "MSFT NVDA AAPL AMZN META"
    seq_length = 60
    
    if Path("best_config.json").exists():
        try:
            with open("best_config.json", "r") as f:
                config = json.load(f)
            
            tickers = " ".join(config.get("tickers", ["MSFT", "NVDA", "AAPL", "AMZN", "META"]))
            seq_length = config.get("seq_length", 60)
            
            print_status("Using configuration from trained model:")
            print(f"   - Tickers: {tickers}")
            print(f"   - Sequence length: {seq_length} days")
        except Exception as e:
            print_warning(f"Could not read best_config.json: {e}")
            print_warning("Using fallback configuration (step 1 defaults)")
    else:
        print_warning("Using fallback configuration (step 1 defaults)")

    # Compute signal generation period (sequence length + buffer)
    signal_required_days = seq_length + 30
    
    if signal_required_days >= 365:
        signal_period = "1y"
    elif signal_required_days >= 180:
        signal_period = "6mo"
    elif signal_required_days >= 90:
        signal_period = "3mo"
    elif signal_required_days >= 60:
        signal_period = "2mo"
    else:
        signal_period = "1mo"
    
    print_status("Computed signal period based on model sequence length:")
    print(f"   - Model sequence length: {seq_length} days")
    print(f"   - Signal period: {signal_period} (requires {signal_required_days} days)")

    args = [
        "--generate-signals",
        "--tickers"] + tickers.split() + [
        "--signals-period", signal_period,
        "--output-dir", "./backtesting_results"
    ]
    
    return_code = run_python_script("run_backtesting.py", args)
    check_previous_step(return_code)
    print_success("Signal generation completed")
    
    return return_code

def setup_cuda_environment():
    """Setup CUDA environment if GPU is available."""
    try:
        # Check if nvidia-smi is available using shutil
        import shutil
        if shutil.which("nvidia-smi"):
            print("🚀 GPU detected - Setting up CUDA library path")
            
            # Try to detect CUDA installation path dynamically
            cuda_lib_path = None
            
            # Method 1: Check common CUDA installation paths
            for cuda_path in ["/usr/local/cuda", "/opt/cuda", "/usr/cuda"]:
                lib_path = Path(cuda_path) / "targets/x86_64-linux/lib"
                if lib_path.exists():
                    cuda_lib_path = str(lib_path)
                    print(f"📁 Found CUDA at: {cuda_path}")
                    break
            
            # Method 2: Use nvcc to find CUDA installation
            if not cuda_lib_path and shutil.which("nvcc"):
                nvcc_path = shutil.which("nvcc")
                if nvcc_path:
                    # nvcc is typically in bin/, so go up to lib/
                    cuda_root = Path(nvcc_path).parent.parent
                    lib_path = cuda_root / "targets/x86_64-linux/lib"
                    if lib_path.exists():
                        cuda_lib_path = str(lib_path)
                        print(f"📁 Found CUDA via nvcc at: {cuda_root}")
            
            # Method 3: Check environment variables
            if not cuda_lib_path and os.environ.get("CUDA_HOME"):
                lib_path = Path(os.environ["CUDA_HOME"]) / "targets/x86_64-linux/lib"
                if lib_path.exists():
                    cuda_lib_path = str(lib_path)
                    print(f"📁 Found CUDA via CUDA_HOME at: {os.environ['CUDA_HOME']}")
            
            # Method 4: Check CUDA_PATH
            if not cuda_lib_path and os.environ.get("CUDA_PATH"):
                lib_path = Path(os.environ["CUDA_PATH"]) / "targets/x86_64-linux/lib"
                if lib_path.exists():
                    cuda_lib_path = str(lib_path)
                    print(f"📁 Found CUDA via CUDA_PATH at: {os.environ['CUDA_PATH']}")
            
            # Set LD_LIBRARY_PATH if CUDA library path was found
            if cuda_lib_path:
                current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
                os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{current_ld_path}"
                print(f"✅ CUDA library path set to: {cuda_lib_path}")
            else:
                print("⚠️  CUDA installation not found - GPU may not work properly")
                print("   Tried paths: /usr/local/cuda, /opt/cuda, /usr/cuda, and environment variables")
        else:
            print("ℹ️  No GPU detected - Running on CPU/TPU")
    except Exception as e:
        print(f"ℹ️  Error checking GPU: {e} - Running on CPU/TPU")

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="Complete Hyperparameter Tuning Pipeline")
    parser.add_argument("--start-step", type=int, default=1, choices=range(1, 7),
                       help="Start from specific step (1-6)")
    parser.add_argument("--no-confirm", action="store_true",
                       help="Run without user confirmation")
    
    args = parser.parse_args()
    
    # Setup CUDA environment
    setup_cuda_environment()
    
    print_header("Starting Complete Hyperparameter Tuning Pipeline")
    print("==================================================")
    
    # Check if we're in the right directory
    if not Path("run_hyperparameter_tuning.py").exists():
        print_error("Please run this script from the project root directory")
        sys.exit(1)
    
    # Check if required directories exist, create if needed
    for dir_name in ["hyperparameter_tuning_results", "extended_training_results", 
                     "backtesting_results", "analysis_plots", "final_model"]:
        Path(dir_name).mkdir(exist_ok=True)
    
    print_status("Environment check completed")
    
    print()
    print("📋 Pipeline Steps:")
    print("1. Hyperparameter Tuning (subset parameters)")
    print("2. Extended Training")
    print("3. Results Analysis")
    print("4. Final Model Training")
    print("5. Backtesting")
    print("6. Signal Generation")
    print()
    
    print_status(f"Starting from step: {args.start_step}")
    if args.no_confirm:
        print_warning("Running without user confirmation")
    print()
    
    # Check prerequisites for starting step
    if not check_step_prerequisites(args.start_step):
        sys.exit(1)
    
    # Run steps based on start step
    if args.start_step == 1:
        run_step_1()
        wait_for_confirmation(args.no_confirm)
        run_step_2()
        wait_for_confirmation(args.no_confirm)
        run_step_3()
        wait_for_confirmation(args.no_confirm)
        run_step_4()
        wait_for_confirmation(args.no_confirm)
        run_step_5()
        wait_for_confirmation(args.no_confirm)
        run_step_6()
    elif args.start_step == 2:
        run_step_2()
        wait_for_confirmation(args.no_confirm)
        run_step_3()
        wait_for_confirmation(args.no_confirm)
        run_step_4()
        wait_for_confirmation(args.no_confirm)
        run_step_5()
        wait_for_confirmation(args.no_confirm)
        run_step_6()
    elif args.start_step == 3:
        run_step_3()
        wait_for_confirmation(args.no_confirm)
        run_step_4()
        wait_for_confirmation(args.no_confirm)
        run_step_5()
        wait_for_confirmation(args.no_confirm)
        run_step_6()
    elif args.start_step == 4:
        run_step_4()
        wait_for_confirmation(args.no_confirm)
        run_step_5()
        wait_for_confirmation(args.no_confirm)
        run_step_6()
    elif args.start_step == 5:
        run_step_5()
        wait_for_confirmation(args.no_confirm)
        run_step_6()
    elif args.start_step == 6:
        run_step_6()
    
    # Final summary
    print()
    print("==================================================")
    print_success("🎉 Pipeline Successfully Completed!")
    print("==================================================")
    
    print()
    print("📁 Generated Files:")
    print("   - Hyperparameter tuning: ./hyperparameter_tuning_results/")
    print("   - Best configuration: ./best_config.json")
    print("   - Analysis plots: ./analysis_plots/")
    print("   - Final model: ./final_model/")
    print("   - Backtesting results: ./backtesting_results/")
    print("   - Trading signals: ./backtesting_results/latest_signals.csv")
    
    print()
    print("🚀 Next Steps:")
    print("   - Review backtesting results in ./backtesting_results/")
    print("   - Check latest signals in ./backtesting_results/latest_signals.csv")
    print("   - Analyze performance plots in ./analysis_plots/")
    print("   - For production, update parameters in this script and re-run")
    print("   - Run step 6 independently for fresh signals: python run_complete_pipeline.py --start-step 6")
    
    print()
    print_success("Pipeline execution completed successfully!")

if __name__ == "__main__":
    main() 