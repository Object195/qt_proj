import pandas as pd
import generate_data
import generate_features
import feature_filter
import create_model_data
import backtest_xgboost
from training import train_xgboost

def run_full_pipeline(fetch_start, fetch_end, train_start, train_end, test_start, test_end):
    # Feature filter start date is dynamically calculated as fetch_start + 256 days
    filter_start = (pd.to_datetime(fetch_start) + pd.Timedelta(days=256)).strftime('%Y-%m-%d')
    filter_end =  fetch_end
    filter_start = '2016-03-01'
    filter_end = '2026-03-01'

    # Create an editable sequence of steps passing the required configurations
    sequence = [
        {'name': 'generate_data', 'func': generate_data.run, 'kwargs': {'fetch_start_date': fetch_start, 'fetch_end_date': fetch_end}},
        {'name': 'generate_features', 'func': generate_features.run, 'kwargs': {}},
        {'name': 'feature_filter', 'func': feature_filter.run, 'kwargs': {'start_date': filter_start, 'end_date':  filter_end }},
        {'name': 'create_model_data', 'func': create_model_data.run, 'kwargs': {
            'train_start': train_start, 'train_end': train_end, 
            'test_start': test_start, 'test_end': test_end, 'model_type': 'xgboost'
        }},
        {'name': 'train_xgboost', 'func': train_xgboost.run, 'kwargs': {}},
        {'name': 'backtest_xgboost', 'func': backtest_xgboost.run, 'kwargs': {
            'train_start': train_start, 'train_end': train_end, 
            'test_start': test_start, 'test_end': test_end
        }}
    ]

    print("=== STARTING PIPELINE ===")
    for step in sequence:
        print(f"\n>>> Executing {step['name']}...")
        step['func'](**step['kwargs'])
    print("\n=== PIPELINE COMPLETE ===")

# Define the core explicit pipeline dates
# (These override any previous implicit config definitions)
FETCH_START = '2015-03-12'
FETCH_END = '2026-03-12'

TRAIN_START = '2021-01-01'
TRAIN_END = '2024-01-01'

TEST_START = '2024-03-01'
TEST_END = '2025-03-01'

if __name__ == '__main__':
    
    # Fire the whole sequence
    run_full_pipeline(FETCH_START, FETCH_END, TRAIN_START, TRAIN_END, TEST_START, TEST_END)