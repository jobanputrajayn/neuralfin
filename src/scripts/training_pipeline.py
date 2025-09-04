"""
Training pipeline utilities for the JAX GPT Stock Predictor.

Contains functions for setting up training environment, data preparation,
model training, and evaluation.
"""

import gc
import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
from etils import epath

from src.models.backtesting import (
    _run_single_backtest,
    combine_equity_curves,
)

# --- Checkpoint Configuration ---
CHECKPOINT_DIR = epath.Path(os.path.abspath("./jax_gpt_stock_predictor_checkpoints"))

# --- Data Caching Configuration ---
DATA_CACHE_DIR = epath.Path(os.path.abspath("./data_cache"))


def perform_backtesting(
    historical_signals,
    tickers,
    all_tickers_data_ohlcv_initial,
    seq_length,
    time_window,
    initial_cash_per_backtest,
    commission_rate,
    report_output_file,
    max_workers: Optional[int] = None,
):
    """
    Performs backtesting for each ticker using pre-computed signals.
    Now uses the optimized combined approach for both training and non-training cases.
    """
    individual_backtest_report_str = (
        "\n--- Individual Ticker Backtesting Performance (Optimized) ---\n"
    )

    all_backtest_stats = {}  # Dictionary to store stats for each ticker
    all_equity_curves = {}  # Dictionary to store equity curves for each ticker

    print("Pre-computing historical signals for backtesting...")

    # Use the new combined approach - handles both training and non-training cases
    historical_signals_for_backtest = historical_signals

    if historical_signals_for_backtest.empty:
        individual_backtest_report_str += (
            "Warning: No historical signals generated. Backtesting will be skipped.\n"
        )
        with open(report_output_file, "a") as f:
            f.write(individual_backtest_report_str)
        return all_equity_curves

    # Determine backtest data period based on signals
    signal_dates = historical_signals_for_backtest.index.get_level_values("Date")
    backtest_start_date = signal_dates.min()
    backtest_end_date = signal_dates.max()

    print(f"Backtesting period: {backtest_start_date} to {backtest_end_date}")

    # Filter data for backtesting period
    backtest_data_period_ohlcv = all_tickers_data_ohlcv_initial.loc[
        backtest_start_date:backtest_end_date
    ]

    if backtest_data_period_ohlcv.empty:
        individual_backtest_report_str += "Warning: No data available for backtesting period. Backtesting will be skipped.\n"
        with open(report_output_file, "a") as f:
            f.write(individual_backtest_report_str)
        return all_equity_curves

    # Prepare ticker data for backtesting
    ticker_data_args = []
    for ticker in tickers:
        ticker_data_for_backtest = None
        if (ticker, "Open") in backtest_data_period_ohlcv.columns:
            ticker_data_for_backtest = (
                backtest_data_period_ohlcv[ticker][
                    ["Open", "High", "Low", "Close", "Volume"]
                ]
                .ffill()
                .dropna()
            )

        if ticker_data_for_backtest is None or ticker_data_for_backtest.empty:
            individual_backtest_report_str += f"  No valid data for backtesting {ticker} in the evaluation period. Skipping.\n"
            continue

        ticker_data_args.append(
            (
                ticker,
                ticker_data_for_backtest,
                historical_signals_for_backtest,
                seq_length,
                time_window,
                initial_cash_per_backtest,
                commission_rate,
            )
        )

    if not ticker_data_args:
        individual_backtest_report_str += (
            "No valid ticker data found for backtesting.\n"
        )
        with open(report_output_file, "a") as f:
            f.write(individual_backtest_report_str)
        return all_equity_curves

    # Determine number of workers
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(ticker_data_args))

    print(
        f"Running backtests for {len(ticker_data_args)} tickers using {max_workers} threads..."
    )

    # Run backtests in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all backtests
        future_to_ticker = {}
        for args in ticker_data_args:
            future = executor.submit(_run_single_backtest, args)
            future_to_ticker[future] = args[0]  # args[0] is the ticker

        # Collect results as they complete
        completed_backtests = 0
        for future in as_completed(future_to_ticker):
            try:
                ticker, stats, equity_curve = future.result()
                if stats is not None and equity_curve is not None:
                    all_backtest_stats[ticker] = stats
                    all_equity_curves[ticker] = equity_curve
                    individual_backtest_report_str += f"\n--- Stats for {ticker} ---\n"
                    individual_backtest_report_str += stats.to_string() + "\n"
                else:
                    individual_backtest_report_str += (
                        f"  Backtest failed for {ticker}.\n"
                    )

                completed_backtests += 1
                if completed_backtests % max(1, len(ticker_data_args) // 10) == 0:
                    print(
                        f"  Completed {completed_backtests}/{len(ticker_data_args)} backtests (threading)..."
                    )

            except Exception as e:
                ticker = future_to_ticker[future]
                print(f"Error in backtest for {ticker}: {e}")
                individual_backtest_report_str += (
                    f"  Error in backtest for {ticker}: {e}\n"
                )
                continue

    with open(report_output_file, "a") as f:
        f.write(individual_backtest_report_str)

    del historical_signals_for_backtest, backtest_data_period_ohlcv
    gc.collect()

    print(
        f"Backtesting complete. Successfully processed {len(all_backtest_stats)} tickers."
    )
    return all_equity_curves


def aggregate_portfolio_performance(
    all_equity_curves, initial_cash_per_backtest, plot, tickers, report_output_file
):
    """
    Aggregates individual equity curves into a single portfolio equity curve
    and calculates overall portfolio statistics.
    """
    portfolio_report_str = (
        "\n--- Aggregated Portfolio Performance from Individual Backtests ---\n"
    )
    if all_equity_curves:
        portfolio_equity_curve_df, portfolio_stats = combine_equity_curves(
            all_equity_curves, initial_cash_per_backtest, tickers
        )
        portfolio_report_str += "Summary Statistics:\n"
        for key, value in portfolio_stats.items():
            if isinstance(value, float):
                portfolio_report_str += f"- {key}: {value:.2f}\n"
            else:
                portfolio_report_str += f"- {key}: {value}\n"
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(
                portfolio_equity_curve_df.index,
                portfolio_equity_curve_df["PortfolioValue"],
                label="Portfolio Value",
                color="blue",
            )
            plt.title(
                "Synthetic Portfolio Equity Curve (Aggregated from Individual Backtests)"
            )
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plot_filename = f"synthetic_portfolio_equity_curve_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
            plt.savefig(plot_filename)
            portfolio_report_str += (
                f"Synthetic portfolio equity curve plot saved to {plot_filename}\n"
            )
            plt.show()
    else:
        portfolio_report_str += "Skipping synthetic portfolio aggregation: No individual equity curves generated.\n"
    with open(report_output_file, "a") as f:
        f.write(portfolio_report_str)
