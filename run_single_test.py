import os
import pandas as pd
import numpy as np
from config import PIPELINE
import prepare_datasets
from training import train_xgboost
from backtest_metrics import BacktestMetrics

def main(verbose=True):
    # Define the date ranges for this specific test run.
    # For rolling window analysis, this dictionary would be updated in a loop.
    date_config = {
        'train_start_date': '2021-01-01',
        'train_end_date': '2024-01-01',
        'test_start_date': '2024-02-01',
        'test_end_date': '2025-02-01',
    }

    if verbose:
        print(f"=== Step 1: Preparing Datasets for XGBoost ===")
    prepare_datasets.main(date_config, model_type='xgboost', verbose=verbose)

    if verbose:
        print(f"\n=== Step 2: Training XGBoost Model ===")
    model = train_xgboost.train_model(verbose=verbose)

    if verbose:
        print(f"\n=== Step 3: Evaluating Test Statistics ===")
    # Load generated test data and original prices
    test_data = np.load('training/datasets/xgboost/test_data.npz', allow_pickle=True)
    X_test, y_test, dates_test = test_data['X'], test_data['y'], test_data['dates']
    
    df_prices = pd.read_csv('processed_data_v2.csv')
    df_prices['ds'] = pd.to_datetime(df_prices['ds'])

    if verbose:
        print(f"Running inference on {len(X_test)} test samples...")
    all_probs = model.predict_proba(X_test)
    predicted_labels = np.argmax(all_probs, axis=1)

    # Compute Adjusted Predictions based on confidence map
    CONFIDENCE_THRESHOLD = 0.5
    adjusted_predicted_labels = np.ones_like(predicted_labels)
    adjusted_predicted_labels[(predicted_labels == 0) & (all_probs[:, 0] > CONFIDENCE_THRESHOLD)] = 0
    adjusted_predicted_labels[(predicted_labels == 2) & (all_probs[:, 2] > CONFIDENCE_THRESHOLD)] = 2

    results_df = pd.DataFrame({
        'ds': pd.to_datetime(dates_test),
        'True_Label': y_test,
        'Predicted_Label': predicted_labels,
        'Adjusted_Predicted_Label': adjusted_predicted_labels
    })

    # Merge inferred labels with prices to evaluate equity curves
    plot_df = pd.merge(results_df, df_prices, on='ds', how='inner').sort_values('ds')
    ndays = PIPELINE.get('forecast_horizon', 5)

    # 1. Print Standard Results
    orig_metrics = BacktestMetrics.calculate_performance(plot_df, 'Predicted_Label', ndays)
    if verbose:
        print("\n--- Original Model Strategy Stats ---")
        print(f"Signal Acc (Δ=0): {orig_metrics['delta_0']:.2%}")
        print(f"Miss (Δ=1):       {orig_metrics['delta_1']:.2%}")
        print(f"Wrong (Δ=2):      {orig_metrics['delta_2']:.2%}")
        print(f"Sim Return ({ndays}d): {orig_metrics['final_return']:.2%}")
        print(f"Sharpe Ratio:     {orig_metrics['sharpe_ratio']:.2f}")

    # 2. Print Adjusted Confidence Results
    adj_metrics = BacktestMetrics.calculate_performance(plot_df, 'Adjusted_Predicted_Label', ndays)
    if verbose:
        print(f"\n--- Adjusted Model Strategy Stats (Conf > {CONFIDENCE_THRESHOLD}) ---")
        print(f"Signal Acc (Δ=0): {adj_metrics['delta_0']:.2%}")
        print(f"Miss (Δ=1):       {adj_metrics['delta_1']:.2%}")
        print(f"Wrong (Δ=2):      {adj_metrics['delta_2']:.2%}")
        print(f"Sim Return ({ndays}d): {adj_metrics['final_return']:.2%}")
        print(f"Sharpe Ratio:     {adj_metrics['sharpe_ratio']:.2f}")
    
    # Distribution output
    adj_pred_dist = plot_df['Adjusted_Predicted_Label'].value_counts(normalize=True).sort_index().to_dict()
    if verbose:
        print(f"\n[Distribution] Adj Predicted Labels: {adj_pred_dist}")


if __name__ == "__main__":
    main()