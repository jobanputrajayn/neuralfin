# Preprocessing, Caching & Restore System - Visual Flow

## Overview
The JAX GPT Stock Predictor uses a sophisticated multi-level caching system to optimize data processing and GPU utilization. Here's how the data flows through the system:

## 1. Initial Data Processing & Caching Flow

```mermaid
graph TD
    A[Raw OHLCV Data from YFinance] --> B[CPU Processing process_and_cache_data]
    B --> C{Check Cache _get_cache_filename}
    C -->|Cache Hit| D[Load from Disk pickle.load]
    C -->|Cache Miss| E[JAX GPU Processing process_ticker_data]
    
    E --> F[GPU Memory all_close_prices_jax]
    F --> G[Parallel Processing jax.vmap]
    G --> H[Processed Data labels, returns, prices]
    H --> I[Save to Disk pickle.dump]
    
    D --> J[Memory Cache processed_data_dict]
    I --> J
    
    J --> K[Lookup Structures data_df, lookup_dict]
    K --> L[Sequence Generator StockSequenceGenerator]
```

## 2. Runtime Sequence Generation Flow

```mermaid
graph TD
    A[Training Request] --> B[StockSequenceGenerator __next__]
    B --> C[Get Batch Indices sequence_indices_to_use]
    C --> D[Generate Sequences _generate_sequence_from_cache]
    
    D --> E[JAX GPU Slicing jax.lax.dynamic_slice]
    E --> F[GPU Memory all_close_prices_jax]
    F --> G[Extract Sequence seq_length x num_tickers]
    
    G --> H[Lookup Labels/Returns lookup_dict]
    H --> I[Create Batch jnp.stack]
    I --> J[Normalize Data scaler_mean/std]
    J --> K[Pad Batch if needed]
    K --> L[Return Batch X, Y, R, mask]
```

## 3. Memory Hierarchy & Data Storage

```mermaid
graph TD
    subgraph "Disk Storage"
        A[data_cache/processed_data_*.pkl]
        B[tensorboard_logs/]
        C[hyperparameter_tuning_results/]
    end
    
    subgraph "CPU Memory"
        D[processed_data_dict]
        E[data_df]
        F[lookup_dict]
        G[sequence_indices_to_use]
    end
    
    subgraph "GPU Memory"
        H[all_close_prices_jax]
        I[batch_x, batch_y, batch_r]
        J[normalized_sequences]
    end
    
    subgraph "Cache Memory"
        K[sequence_cache]
        L[prefetch_buffer]
    end
    
    A --> D
    D --> E
    D --> F
    E --> H
    F --> I
    G --> I
    I --> J
    J --> K
    K --> L
```

## 4. Prefetching & GPU Optimization Flow

```mermaid
graph TD
    A[PrefetchGenerator] --> B[Background Thread _fill_buffer]
    B --> C[GPU Manager get_gpu_manager]
    C --> D[Transfer to GPU to_device]
    D --> E[Queue Buffer maxsize=5]
    
    E --> F[Training Loop __next__]
    F --> G{Buffer Empty?}
    G -->|Yes| H[Direct Transfer to_device]
    G -->|No| I[Get from Buffer cache_hit]
    
    H --> J[GPU Processing Model Forward Pass]
    I --> J
```

## 5. Adaptive Batch Size & Memory Management

```mermaid
graph TD
    A[Batch Generation] --> B[Time Measurement batch_start_time]
    B --> C[Generate Batch _generate_sequence_from_cache]
    C --> D[Calculate Batch Time time.time - start_time]
    
    D --> E{Adaptive Batch Size?}
    E -->|Yes| F[Adjust Batch Size _adjust_batch_size]
    E -->|No| G[Return Batch]
    
    F --> H{Detect JAX Recompilation?}
    H -->|Yes| I[Conservative Adjustment 0.7x batch size]
    H -->|No| J[Performance-based Adjustment exponential scaling]
    
    I --> K[Apply Memory Constraints _apply_jax_memory_constraints]
    J --> K
    K --> L[Update Batch Size self batch size]
    L --> G
```

## 6. Data Flow Architecture

```mermaid
graph LR
    subgraph "Input Layer"
        A[YFinance Data]
        B[OHLCV DataFrame]
    end
    
    subgraph "Processing Layer"
        C[CPU Processing]
        D[JAX GPU Processing]
        E[Cache Management]
    end
    
    subgraph "Storage Layer"
        F[Disk Cache]
        G[Memory Cache]
        H[GPU Memory]
    end
    
    subgraph "Output Layer"
        I[Sequence Generator]
        J[Prefetch Buffer]
        K[Training Batches]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    E --> G
    E --> H
    F --> I
    G --> I
    H --> I
    I --> J
    J --> K
```

## 7. Memory Usage Patterns

```mermaid
graph TD
    A[Memory Monitoring] --> B[get_memory_usage]
    B --> C[Cache Size len cache]
    B --> D[Price Data Memory all_close_prices_jax nbytes]
    B --> E[Total Memory cache plus price_data]
    
    E --> F{Memory > Limit?}
    F -->|Yes| G[optimize_for_memory]
    F -->|No| H[Continue Normal Operation]
    
    G --> I[Reduce Cache Size target cache size]
    G --> J[Reduce Batch Size batch size times 0.8]
    I --> K[Update Configuration]
    J --> K
```

## 8. Cache Hit/Miss Statistics

```mermaid
graph TD
    A[Cache Request] --> B{Key in Cache?}
    B -->|Yes| C[Cache Hit stats cache_hits increment]
    B -->|No| D[Cache Miss stats cache_misses increment]
    
    C --> E[Return Cached Data]
    D --> F[Generate New Data]
    F --> G[Store in Cache]
    G --> H[LRU Eviction if cache is full]
    
    E --> I[Training]
    F --> I
    H --> I
```

## Key Performance Optimizations

### 1. **Multi-Level Caching**
- **Disk Cache**: Persistent storage of processed data
- **Memory Cache**: Fast access to frequently used sequences
- **GPU Memory**: Pre-allocated arrays for batch processing

### 2. **JAX GPU Acceleration**
- **JIT Compilation**: `@partial(jax.jit, static_argnums=(1, 2, 3))`
- **Vectorized Operations**: `jax.vmap(process_time_point)`
- **Memory-Efficient Slicing**: `jax.lax.dynamic_slice`

### 3. **Adaptive Batch Sizing**
- **Performance Monitoring**: Track batch generation times
- **JAX Recompilation Detection**: Sudden 2x time increase
- **Memory Pressure Response**: Reduce batch size under memory constraints

### 4. **Prefetching System**
- **Background Threading**: Load data while GPU processes current batch
- **Queue Management**: Buffer multiple batches ahead
- **GPU Transfer Optimization**: Minimize data transfer overhead

## Memory Usage Breakdown

| Component | Memory Type | Size | Purpose |
|-----------|-------------|------|---------|
| `all_close_prices_jax` | GPU Memory | ~100MB | All ticker price data |
| `data_df` | CPU Memory | ~50MB | Processed data lookup |
| `lookup_dict` | CPU Memory | ~10MB | Fast label/return access |
| `sequence_cache` | CPU Memory | Variable | Frequently used sequences |
| `prefetch_buffer` | GPU Memory | ~50MB | Pre-loaded batches |

## Performance Metrics

- **Cache Hit Rate**: Typically 80-95% after warmup
- **Batch Generation Time**: 0.01-0.05 seconds per batch
- **GPU Memory Utilization**: 60-80% during training
- **Disk I/O**: Minimal after initial cache creation

This system ensures optimal GPU utilization while maintaining fast data access and minimal memory overhead. 