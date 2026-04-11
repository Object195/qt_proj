import os
import shutil
import pandas as pd
from tqdm import tqdm
import generate_data
import generate_features
import feature_filter
import create_model_data
from training import train_xgboost

def parse_interval(interval_str):
    """Helper to consistently convert intervals (like 90D or 3M) into Timedeltas or DateOffsets."""
    if isinstance(interval_str, str) and interval_str.upper().endswith('M'):
        return pd.DateOffset(months=int(interval_str[:-1]))
    return pd.Timedelta(interval_str)

def run_sliding_window(
    fetch_start, fetch_end,
    window_start, window_end,
    training_length, test_length, gap,
    run_preparation=True,
    verbose=False
):
    # 1. First Sequence: Run data preparations (One-Time Execution)
    if run_preparation:
        filter_start = '2016-03-01'
        filter_end = '2026-03-01'
        
        prep_sequence = [
            {'name': 'generate_data', 'func': generate_data.run, 'kwargs': {'fetch_start_date': fetch_start, 'fetch_end_date': fetch_end}},
            {'name': 'generate_features', 'func': generate_features.run, 'kwargs': {}},
            {'name': 'feature_filter', 'func': feature_filter.run, 'kwargs': {'start_date': filter_start, 'end_date': filter_end}}
        ]
        
        print("=== STARTING DATA PREPARATION ===")
        for step in prep_sequence:
            print(f"\n>>> Executing {step['name']}...")
            step['func'](**step['kwargs'])
        print("\n=== DATA PREPARATION COMPLETE ===")

    # 2. Second Sequence: Run iterative models
    print("\n=== STARTING SLIDING WINDOW LOOP ===")
    print("Specifications:")
    print(f"  Window Start : {window_start}")
    print(f"  Window End   : {window_end}")
    print(f"  Train Length : {training_length}")
    print(f"  Test Length  : {test_length}")
    print(f"  Gap          : {gap}")
    
    # Store sliding window outputs in this directory
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sliding_window_results')
    
    # Overwrite previous runs to save space, assuming backup is done manually
    if os.path.exists(base_dir):
        if verbose: print(f"Removing existing sliding window results at {base_dir}...")
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)
    
    current_start = pd.to_datetime(window_start)
    end_date = pd.to_datetime(window_end)
    
    train_td = parse_interval(training_length)
    test_td = parse_interval(test_length)
    gap_td = parse_interval(gap)
    
    # Pre-calculate total iterations for progress bar
    temp_start = current_start
    total_iters = 0
    while True:
        t_train_end = temp_start + train_td
        t_test_start = t_train_end + gap_td
        if t_test_start >= end_date:
            break
        total_iters += 1
        temp_start += test_td
        
    print(f"  Total Iters  : {total_iters}\n")

    iteration = 1
    
    with tqdm(total=total_iters, desc="Overall Progress") as pbar:
        while True:
            train_start = current_start
            train_end = train_start + train_td
            test_start = train_end + gap_td
            test_end = test_start + test_td
            
            # Break if we shift beyond the total window end bounds
            if test_start >= end_date:
                break
                
            # Cap the final test sequence to not exceed the dataset bound 
            if test_end > end_date:
                test_end = end_date
                
            if verbose:
                print(f"\n--- Iteration {iteration} ---")
                print(f"Train: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
                print(f"Test:  {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
            
            iter_dir = os.path.join(base_dir, f"iter_{iteration}")
            data_dir = os.path.join(iter_dir, 'data')
            model_dir = os.path.join(iter_dir, 'model')
            
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)
            
            if verbose: print(f"\n>>> Executing create_model_data for iteration {iteration}...")
            create_model_data.run(
                train_start=train_start.strftime('%Y-%m-%d'),
                train_end=train_end.strftime('%Y-%m-%d'),
                test_start=test_start.strftime('%Y-%m-%d'),
                test_end=test_end.strftime('%Y-%m-%d'),
                model_type='xgboost',
                output_dir=data_dir,
                verbose=verbose
            )
            
            if verbose: print(f"\n>>> Executing train_xgboost for iteration {iteration}...")
            train_xgboost.run(
                data_dir=data_dir,
                model_save_dir=model_dir,
                verbose=verbose
            )
            
            # Shift start point forward by length of the test segment
            current_start += test_td
            iteration += 1
            pbar.update(1)

    print("\n=== SLIDING WINDOW LOOP COMPLETE ===")
    print(f"All models and datasets saved in: {base_dir}")

if __name__ == '__main__':
    FETCH_START = '2015-03-12'
    FETCH_END = '2026-03-12'
    
    WINDOW_START = '2015-01-01'
    WINDOW_END = '2025-01-01'
    
    TRAINING_LENGTH = '36M'   # 3 years
    TEST_LENGTH = '6M'        # 3 months
    GAP = '1M'                # 1 month gap
    
    # Toggle run_preparation to True to regenerate the full dataset and features initially 
    run_sliding_window(
        fetch_start=FETCH_START, fetch_end=FETCH_END,
        window_start=WINDOW_START, window_end=WINDOW_END,
        training_length=TRAINING_LENGTH, test_length=TEST_LENGTH, gap=GAP,
        run_preparation=False 
    )