"""
Backtesting functionality for the JAX GPT Stock Predictor.

Contains the GPTOptionsStrategy class and related backtesting utilities.
"""

import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
from typing import Dict, List, Optional, Any
import jax
import jax.numpy as jnp
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from flax import nnx

from .constants import ACTION_HOLD, ACTION_BUY_CALL, ACTION_BUY_PUT, NUM_CLASSES
from src.utils.system_utils import calculate_financial_metrics
from src.data.sequence_generator import StockSequenceGenerator  # <-- Add this import


class GPTOptionsStrategy(Strategy):
    """
    A backtesting strategy that uses pre-computed, universe-wide signals to make trading decisions.
    """
    # Parameters to be passed during Backtest initialization
    historical_signals_df = None  # DataFrame with pre-computed signals
    seq_length = 100
    time_window = 22
    profit_target_percent = 0.02
    stop_loss_percent = 0.10
    symbol = None

    def init(self):
        """Initializes the strategy, called once at the beginning of backtest."""
        self.trade_entry_bar = None
        self.entry_price = None

    def next(self):
        """
        Called on each bar (time step) of the backtest.
        Decides whether to enter or exit a trade based on pre-computed signals.
        """
        current_price = self.data.Close[-1]
        current_date = self.data.index[-1]

        # --- Exit Logic: Check for profit target or stop loss ---
        if self.position:
            # Calculate current profit/loss
            if self.position.is_long:
                current_pnl = (current_price - self.entry_price) / self.entry_price
            else:  # Short position
                current_pnl = (self.entry_price - current_price) / self.entry_price

            # Check profit target
            if current_pnl >= self.profit_target_percent:
                self.position.close()
                # print(f"Profit target hit for {self.symbol} at {current_date.strftime('%Y-%m-%d')} @ {current_price:.4f} (P&L: {current_pnl:.2%})")
                return

            # Check stop loss
            if current_pnl <= -self.stop_loss_percent:
                self.position.close()
                # print(f"Stop loss hit for {self.symbol} at {current_date.strftime('%Y-%m-%d')} @ {current_price:.4f} (P&L: {current_pnl:.2%})")
                return

        # --- Entry Logic: Enter a new position if no current position and signal exists ---
        if not self.position:
            decision_str = None
            # Ensure current_date is a proper Timestamp object
            current_date_ts = pd.to_datetime(self.data.index[-1])

            # Skip if current_date is not a valid date
            if pd.isna(current_date_ts):
                return  # Skip this iteration

            key_tuple = (self.symbol, current_date_ts)

            try:
                # Attempt to retrieve the signal using .loc for MultiIndex lookup
                decision_str = self.historical_signals_df.loc[key_tuple, 'Predicted_Action']
            except KeyError:
                # This is expected if a signal doesn't exist for that specific ticker/date combination
                decision_str = None
            except Exception as e:
                # Catch any other unexpected errors during lookup for debugging
                print(f"Error looking up signal for {self.symbol} on {current_date_ts.strftime('%Y-%m-%d')}: {e}")
                decision_str = None  # Ensure decision_str is None on error

            if decision_str == 'BUY_CALL':
                self.buy()
                self.trade_entry_bar = len(self.data)
                self.entry_price = current_price
                # print(f"Entered LONG {self.symbol} at {current_date.strftime('%Y-%m-%d')} @ {current_price:.4f}")
            elif decision_str == 'BUY_PUT':
                self.sell()
                self.trade_entry_bar = len(self.data)
                self.entry_price = current_price
                # print(f"Entered SHORT {self.symbol} at {current_date.strftime('%Y-%m-%d')} @ {current_price:.4f}")


def precompute_historical_signals(
    all_tickers_data_ohlcv,
    model_def,
    scaler_mean,
    scaler_std,
    seq_length,
    num_input_features,
    backtest_start_date=None,
    max_workers=None,
    include_news=False,
    include_text=False,
    news_window=7,
    text_model='sentence-transformers/all-MiniLM-L6-v2',
    tickers=None,
    latest_only=False,
    time_window=1,
    batch_size=64
):
    """
    Pre-computes historical signals for all tickers using the trained model and StockSequenceGenerator.
    Supports enhanced features (news, text) if enabled.
    If latest_only is True, only the latest sequence is generated (for signal generation).
    Honors the time_window parameter for sequence generation and label/return calculation.
    """
    print(f"=== Precompute Historical Signals (with Sequence Generator) ===")
    print(f"Data shape: {all_tickers_data_ohlcv.shape}")
    print(f"Sequence length: {seq_length}")
    print(f"Number of input features: {num_input_features}")
    print(f"Enhanced features: news={include_news}, text={include_text}, news_window={news_window}")
    print(f"Available data points: {len(all_tickers_data_ohlcv.index)}")
    print(f"Date range: {all_tickers_data_ohlcv.index.min()} to {all_tickers_data_ohlcv.index.max()}")

    if tickers is None:
        tickers = all_tickers_data_ohlcv.columns.get_level_values(0).unique().tolist()
    
    dates = all_tickers_data_ohlcv.index
    if latest_only:
        # Only generate the latest valid sequence
        if len(dates) < seq_length:
            print(f"Not enough data for latest sequence: {len(dates)} < {seq_length}")
            return pd.DataFrame(columns=['Ticker', 'Date', 'Predicted_Action'])
        prediction_indices = [len(dates) - seq_length]
        prediction_dates = [dates[-1]]
    else:
        if backtest_start_date is None:
            prediction_start_idx = seq_length - 1
            required_start_idx = 0
            required_start_date = dates[0]
            if prediction_start_idx >= len(dates):
                print(f"Error: prediction_start_idx ({prediction_start_idx}) >= len(dates) ({len(dates)})")
                return pd.DataFrame(columns=['Ticker', 'Date', 'Predicted_Action'])
            prediction_start_date = dates[prediction_start_idx]
        else:
            required_start_date = backtest_start_date - pd.Timedelta(days=seq_length-1)
            prediction_start_date = backtest_start_date
            try:
                required_start_idx = dates.get_loc(required_start_date)
                prediction_start_idx = dates.get_loc(prediction_start_date)
            except KeyError as e:
                print(f"Date not found in index: {e}")
                return pd.DataFrame(columns=['Ticker', 'Date', 'Predicted_Action'])
        all_indices = list(range(required_start_idx, len(dates) - seq_length + 1))
        # Robust fix: Ensure lower bound is not greater than upper bound and indices are valid
        lower = max(0, required_start_idx)
        upper = len(dates) - (seq_length + time_window ) + 1
        if lower >= upper:
            print(f"No valid prediction indices: lower={lower}, upper={upper}, len(dates)={len(dates)}, seq_length={seq_length}")
            return pd.DataFrame(columns=['Ticker', 'Date', 'Predicted_Action'])
        prediction_indices = [i for i in range(lower, upper)]
        prediction_dates = dates[prediction_start_idx:]

    generator = StockSequenceGenerator(
        sequence_indices_to_use=prediction_indices,
        all_ohlcv_data=all_tickers_data_ohlcv,
        seq_length=seq_length,
        time_window=time_window,  # <-- use the provided time_window
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        batch_size=batch_size,  # Use a reasonable batch size for inference
        shuffle_indices=False,
        tickers=tickers,
        enable_caching=False,
        include_news=include_news,
        include_text=include_text,
        news_window=news_window,
        text_model=text_model,
        inference_mode=latest_only
    )

    # Add a jitted eval_step for SPMD/multi-device evaluation
    @nnx.jit
    def eval_step(model, batch_x, padding_mask):
        return model(batch_x, padding_mask=padding_mask)

    action_map = {ACTION_HOLD: 'HOLD', ACTION_BUY_CALL: 'BUY_CALL', ACTION_BUY_PUT: 'BUY_PUT'}
    all_results = []
    model_def.eval()

    prediction_pointer = 0
    for batch in generator:
        batch_x, _, _, padding_mask = batch
        logits = eval_step(model_def, batch_x, padding_mask)
        decisions = jnp.argmax(logits, axis=-1)  # (batch_size, num_tickers)
        valid_indices = np.where(np.array(padding_mask))[0]
        batch_pred_indices = prediction_indices[prediction_pointer:prediction_pointer + len(valid_indices)]
        for idx_in_batch, i in enumerate(valid_indices):
            pred_idx = batch_pred_indices[idx_in_batch]
            if latest_only:
                pred_idx = 0
            if pred_idx < 0 or pred_idx >= len(prediction_dates):
                continue
            pred_date = prediction_dates[pred_idx]
            for j, target_ticker in enumerate(tickers):
                predicted_action_enum = decisions[i, j].item()
                predicted_action_str = action_map.get(predicted_action_enum, 'UNKNOWN')
                all_results.append({
                    'Ticker': target_ticker,
                    'Date': pred_date,
                    'Predicted_Action': predicted_action_str
                })
        prediction_pointer += len(valid_indices)

    if not all_results:
        print("No signals were generated during processing.")
        return pd.DataFrame(columns=['Ticker', 'Date', 'Predicted_Action'])

    historical_signals_df = pd.DataFrame(all_results)
    historical_signals_df['Predicted_Action'] = historical_signals_df['Predicted_Action'].astype('category')
    historical_signals_df = historical_signals_df.set_index(['Ticker', 'Date'])
    historical_signals_df = historical_signals_df.sort_index(level=['Ticker', 'Date'])

    print(f"Pre-computation complete. Generated {len(historical_signals_df)} historical signals.")
    print(f"Date range: {historical_signals_df.index.get_level_values('Date').min()} to {historical_signals_df.index.get_level_values('Date').max()}")
    return historical_signals_df


def combine_equity_curves(individual_equity_curves, initial_capital_per_ticker, tickers):
    """
    Combines individual equity curves into a single portfolio equity curve.
    Assumes each equity curve starts with 'Initial Capital' as the first value.

    Args:
        individual_equity_curves (dict): A dictionary where keys are ticker symbols
                                         and values are pandas Series/DataFrames
                                         representing the equity curve (e.e., stats._equity_curve).
        initial_capital_per_ticker (float): The initial capital used for each individual backtest.

    Returns:
        pd.DataFrame: A DataFrame with the combined portfolio equity curve and daily returns.
        dict: A dictionary of calculated portfolio statistics.
    """
    if not individual_equity_curves:
        print("No individual equity curves to combine.")
        return pd.DataFrame(), {}

    # Standardize and normalize individual equity curves
    normalized_curves = {}
    for ticker in tickers:  # Iterate through provided tickers list to preserve order
        if ticker in individual_equity_curves:
            curve = individual_equity_curves[ticker]
            if isinstance(curve, pd.Series):
                equity_values = curve
            elif isinstance(curve, pd.DataFrame) and 'Equity' in curve.columns:
                equity_values = curve['Equity']
            else:
                print(f"Skipping {ticker}: Equity curve not in expected format.")
                continue
            if len(equity_values) > 0 and initial_capital_per_ticker != 0:
                normalized_curves[ticker] = equity_values / initial_capital_per_ticker
            else:
                normalized_curves[ticker] = pd.Series(1.0, index=equity_values.index)
        else:
            print(f"Info: No equity curve found for ticker {ticker}. Skipping for portfolio aggregation.")
    
    if not normalized_curves:
        print("No valid normalized equity curves to combine.")
        return pd.DataFrame(), {}

    combined_normalized_df = pd.concat([normalized_curves[t] for t in tickers if t in normalized_curves], axis=1, join='outer', keys=[t for t in tickers if t in normalized_curves])
    combined_normalized_df = combined_normalized_df.ffill().bfill()
    num_tickers_backtested = len(normalized_curves)
    if num_tickers_backtested == 0:
        print("No tickers effectively backtested to combine.")
        return pd.DataFrame(), {}
    
    total_initial_capital = initial_capital_per_ticker * num_tickers_backtested
    portfolio_equity_curve = combined_normalized_df.mean(axis=1) * initial_capital_per_ticker
    portfolio_df = pd.DataFrame({'PortfolioValue': portfolio_equity_curve})
    portfolio_df['DailyReturn'] = portfolio_df['PortfolioValue'].pct_change()
    
    # Use the new comprehensive financial metrics calculation
    portfolio_stats = calculate_financial_metrics(
        portfolio_df['PortfolioValue'],
        risk_free_rate=0.02,  # 2% annual risk-free rate
        trading_days_per_year=252
    )
    
    # Add additional portfolio-specific metrics
    portfolio_stats.update({
        'Initial Capital': initial_capital_per_ticker,
        'Final Portfolio Value': portfolio_df['PortfolioValue'].iloc[-1],
        'Total Return [%]': portfolio_stats['total_return'] * 100,
        'Max. Drawdown [%]': portfolio_stats['max_drawdown'] * 100,
        'Annualized Return [%]': portfolio_stats['annualized_return'] * 100,
        'Annualized Volatility [%]': portfolio_stats['annualized_volatility'] * 100,
        'Sharpe Ratio': portfolio_stats['sharpe_ratio'],
        'Calmar Ratio': portfolio_stats['calmar_ratio'],
        'Sortino Ratio': portfolio_stats['sortino_ratio'],
        'VaR (95%)': portfolio_stats['var_95'] * 100,
        'CVaR (95%)': portfolio_stats['cvar_95'] * 100 if not pd.isna(portfolio_stats['cvar_95']) else np.nan
    })
    
    return portfolio_df, portfolio_stats


def _run_single_backtest(args):
    """
    Helper function to run a single ticker backtest.
    This function is designed to be called by ProcessPoolExecutor.
    """
    (ticker, ticker_data_for_backtest, historical_signals_for_backtest, seq_length, 
     time_window, initial_cash_per_backtest, commission_rate) = args
    
    try:
        if not isinstance(ticker_data_for_backtest.index, pd.DatetimeIndex):
            ticker_data_for_backtest.index = pd.to_datetime(ticker_data_for_backtest.index)

        bt = Backtest(ticker_data_for_backtest, GPTOptionsStrategy, 
                     cash=initial_cash_per_backtest, commission=commission_rate)

        stats = bt.run(
            historical_signals_df=historical_signals_for_backtest,
            seq_length=seq_length,
            time_window=time_window,
            profit_target_percent=GPTOptionsStrategy.profit_target_percent,
            stop_loss_percent=GPTOptionsStrategy.stop_loss_percent,
            symbol=ticker
        )
        
        return ticker, stats, stats._equity_curve
    except Exception as e:
        print(f"Error in backtest for {ticker}: {e}")
        return ticker, None, None 