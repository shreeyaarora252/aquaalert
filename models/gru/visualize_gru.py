"""
Visualization script for GRU Water Stress Index Forecasting Model
Creates state-wise plots of predicted vs actual WSI values
"""
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import pickle
from gru_model import GRUForecaster
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages


def load_model_and_scalers(model_path, scalers_path, feature_cols_path, split_info_path, device='cpu'):
    """Load trained model, scalers, feature columns, and split information"""
    with open(feature_cols_path, 'rb') as f:
        feature_cols = pickle.load(f)
    
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    
    # Try to load split_info, if it doesn't exist, we'll recreate it
    try:
        with open(split_info_path, 'rb') as f:
            split_info = pickle.load(f)
        print(f"Loaded split information from {split_info_path}")
    except FileNotFoundError:
        print(f"Warning: Split info file not found at {split_info_path}")
        print("This likely means the model was trained with old code.")
        print("Split info will be recreated using the same 80-20 logic.")
        split_info = None  # Will be recreated in get_state_predictions
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = GRUForecaster(
        input_size=len(feature_cols) + 1,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        output_size=1
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    return model, scalers, feature_cols, split_info


def create_sequences(data, sequence_length, target_col_idx):
    """Create sequences for time series forecasting"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, target_col_idx])
    return np.array(X), np.array(y).reshape(-1, 1)


def get_state_predictions(df, model, scalers, feature_cols, split_info, target_column, 
                          sequence_length=12, train_split_ratio=0.8, device='cpu'):
    """
    Get predictions for each state separately using the same 80-20 split as training
    
    Returns:
        Dictionary with state names as keys and DataFrames with predictions as values
    """
    df = df.sort_values(['state', 'year', 'month']).reset_index(drop=True)
    
    state_results = {}
    
    for state in df['state'].unique():
        state_df = df[df['state'] == state].copy()
        
        if state not in scalers:
            continue
        
        # If split_info is None, we'll recreate it (for backward compatibility)
        if split_info is not None and state not in split_info:
            continue
        
        if len(state_df) < sequence_length + 1:
            continue
        
        # Use the same split as training (80-20 split)
        split_idx = int(len(state_df) * train_split_ratio)
        
        if split_idx < sequence_length + 1:
            continue
        
        train_df = state_df.iloc[:split_idx].copy()
        test_df = state_df.iloc[split_idx:].copy()
        
        if len(test_df) == 0 or len(train_df) < sequence_length:
            continue
        
        scaler = scalers[state]
        
        # Prepare test data
        test_features = test_df[feature_cols].values
        test_target = test_df[target_column].values.reshape(-1, 1)
        test_combined = np.hstack([test_features, test_target])
        test_scaled = scaler.transform(test_combined)
        
        target_col_idx = test_scaled.shape[1] - 1
        
        # Create sequences from test data
        if len(train_df) >= sequence_length:
            train_features = train_df[feature_cols].values
            train_target = train_df[target_column].values.reshape(-1, 1)
            train_combined = np.hstack([train_features, train_target])
            train_scaled = scaler.transform(train_combined)
            combined_scaled = np.vstack([train_scaled[-sequence_length:], test_scaled])
        else:
            combined_scaled = test_scaled
        
        X_test, y_test = create_sequences(combined_scaled, sequence_length, target_col_idx)
        
        if len(X_test) == 0:
            continue
        
        # Make predictions
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        
        with torch.no_grad():
            predictions = model(X_test_tensor).cpu().numpy()
        
        # Inverse transform predictions and actuals
        predictions_unscaled = []
        actuals_unscaled = []
        
        for i in range(len(predictions)):
            # Inverse transform prediction
            dummy_row = combined_scaled[i + sequence_length].copy()
            dummy_row[target_col_idx] = predictions[i, 0]
            pred_unscaled = scaler.inverse_transform(dummy_row.reshape(1, -1))
            predictions_unscaled.append(pred_unscaled[0, target_col_idx])
            
            # Inverse transform actual
            actual_unscaled = scaler.inverse_transform(combined_scaled[i + sequence_length].reshape(1, -1))
            actuals_unscaled.append(actual_unscaled[0, target_col_idx])
        
        # Get corresponding dates from the ACTUAL test set (last 20%)
        # We need to map predictions back to the correct test dates
        # Each prediction corresponds to a test data point
        num_predictions = len(predictions_unscaled)
        test_dates = test_df.iloc[:num_predictions][['year', 'month']].copy()
        
        # Verify we're using the correct test set dates
        test_start_date = f"{test_df.iloc[0]['year']}-{int(test_df.iloc[0]['month']):02d}"
        test_end_date = f"{test_df.iloc[-1]['year']}-{int(test_df.iloc[-1]['month']):02d}"
        train_end_date = f"{train_df.iloc[-1]['year']}-{int(train_df.iloc[-1]['month']):02d}"
        
        # Create results DataFrame with actual test set dates
        results_df = pd.DataFrame({
            'year': test_dates['year'].values,
            'month': test_dates['month'].values,
            'actual': actuals_unscaled,
            'predicted': predictions_unscaled
        })
        
        # Calculate metrics
        mse = mean_squared_error(actuals_unscaled, predictions_unscaled)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals_unscaled, predictions_unscaled)
        r2 = r2_score(actuals_unscaled, predictions_unscaled)
        
        state_results[state] = {
            'data': results_df,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'num_samples': len(predictions_unscaled)
            },
            'split_info': {
                'train_end': train_end_date,
                'test_start': test_start_date,
                'test_end': test_end_date,
                'train_samples': len(train_df),
                'test_samples': len(test_df)
            }
        }
    
    return state_results


def plot_state_comparison(state_results, output_path, index_name='', max_states_per_page=12):
    """
    Create comparison plots for all states
    
    Args:
        state_results: Dictionary with state results
        output_path: Path to save the plot
        index_name: Name of the WSI index
        max_states_per_page: Maximum number of states to plot per page
    """
    states = sorted(state_results.keys())
    num_states = len(states)
    
    # Calculate number of pages needed
    num_pages = (num_states + max_states_per_page - 1) // max_states_per_page
    
    with PdfPages(output_path) as pdf:
        for page in range(num_pages):
            start_idx = page * max_states_per_page
            end_idx = min(start_idx + max_states_per_page, num_states)
            page_states = states[start_idx:end_idx]
            
            # Calculate grid dimensions
            num_cols = 4
            num_rows = (len(page_states) + num_cols - 1) // num_cols
            
            fig = plt.figure(figsize=(16, 4 * num_rows))
            gs = gridspec.GridSpec(num_rows, num_cols, figure=fig, hspace=0.3, wspace=0.3)
            
            fig.suptitle(f'Predicted vs Actual WSI by State - {index_name.upper()} WSI (Page {page + 1}/{num_pages})',
                        fontsize=16, fontweight='bold')
            
            for idx, state in enumerate(page_states):
                row = idx // num_cols
                col = idx % num_cols
                ax = fig.add_subplot(gs[row, col])
                
                data = state_results[state]['data']
                metrics = state_results[state]['metrics']
                
                # Scatter plot
                ax.scatter(data['actual'], data['predicted'], alpha=0.6, s=50)
                
                # Perfect prediction line
                min_val = min(data['actual'].min(), data['predicted'].min())
                max_val = max(data['actual'].max(), data['predicted'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='Perfect Prediction')
                
                # Labels and title
                ax.set_xlabel('Actual WSI', fontsize=9)
                ax.set_ylabel('Predicted WSI', fontsize=9)
                # Add date range to title to show it's the test set
                split_info = state_results[state].get('split_info', {})
                date_range = ""
                if split_info:
                    date_range = f"\nTest: {split_info.get('test_start', '')} to {split_info.get('test_end', '')}"
                ax.set_title(f'{state}{date_range}\nR²={metrics["r2"]:.3f}, RMSE={metrics["rmse"]:.2f}', 
                            fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                
                # Set equal aspect ratio
                ax.set_aspect('auto')
            
            # Hide empty subplots
            for idx in range(len(page_states), num_rows * num_cols):
                row = idx // num_cols
                col = idx % num_cols
                fig.delaxes(fig.add_subplot(gs[row, col]))
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    print(f"State comparison plots saved to {output_path}")


def plot_state_timeseries(state_results, output_path, index_name='', max_states_per_page=6):
    """
    Create time series plots showing actual vs predicted over time for each state
    
    Args:
        state_results: Dictionary with state results
        output_path: Path to save the plot
        index_name: Name of the WSI index
        max_states_per_page: Maximum number of states to plot per page
    """
    states = sorted(state_results.keys())
    num_states = len(states)
    
    # Calculate number of pages needed
    num_pages = (num_states + max_states_per_page - 1) // max_states_per_page
    
    with PdfPages(output_path) as pdf:
        for page in range(num_pages):
            start_idx = page * max_states_per_page
            end_idx = min(start_idx + max_states_per_page, num_states)
            page_states = states[start_idx:end_idx]
            
            # Calculate grid dimensions
            num_cols = 2
            num_rows = (len(page_states) + num_cols - 1) // num_cols
            
            fig = plt.figure(figsize=(16, 4 * num_rows))
            gs = gridspec.GridSpec(num_rows, num_cols, figure=fig, hspace=0.4, wspace=0.3)
            
            fig.suptitle(f'Time Series: Actual vs Predicted WSI - {index_name.upper()} WSI (Page {page + 1}/{num_pages})',
                        fontsize=16, fontweight='bold')
            
            for idx, state in enumerate(page_states):
                row = idx // num_cols
                col = idx % num_cols
                ax = fig.add_subplot(gs[row, col])
                
                data = state_results[state]['data']
                metrics = state_results[state]['metrics']
                
                # Create date index for plotting
                data_sorted = data.sort_values(['year', 'month']).copy()
                data_sorted['date'] = pd.to_datetime(
                    data_sorted['year'].astype(str) + '-' + 
                    data_sorted['month'].astype(str).str.zfill(2) + '-01'
                )
                
                # Plot lines
                ax.plot(data_sorted['date'], data_sorted['actual'], 
                       'o-', label='Actual', linewidth=2, markersize=4, alpha=0.7)
                ax.plot(data_sorted['date'], data_sorted['predicted'], 
                       's-', label='Predicted', linewidth=2, markersize=4, alpha=0.7)
                
                # Labels and title
                ax.set_xlabel('Date', fontsize=10)
                ax.set_ylabel('WSI', fontsize=10)
                # Add date range info to show it's the test set
                split_info = state_results[state].get('split_info', {})
                date_info = ""
                if split_info:
                    date_info = f" (Test: {split_info.get('test_start', '')} to {split_info.get('test_end', '')})"
                ax.set_title(f'{state}{date_info}\nR²={metrics["r2"]:.3f}, RMSE={metrics["rmse"]:.2f}, MAE={metrics["mae"]:.2f}',
                            fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9)
                
                # Rotate x-axis labels
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Hide empty subplots
            for idx in range(len(page_states), num_rows * num_cols):
                row = idx // num_cols
                col = idx % num_cols
                fig.delaxes(fig.add_subplot(gs[row, col]))
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    print(f"Time series plots saved to {output_path}")


def visualize_single_index(df, index_name, target_column, script_path,
                           SEQUENCE_LENGTH=12, TRAIN_SPLIT_RATIO=0.8, device='cpu'):
    """
    Create visualizations for a single WSI index using the same 80-20 split as training
    
    Args:
        df: Full DataFrame
        index_name: Name of the index ('entropy' or 'equal')
        target_column: Target column name
        script_path: Path object for file paths
        TRAIN_SPLIT_RATIO: Ratio used for training (0.8 = 80% train, 20% test)
        Other args: Model parameters
    """
    print(f"\n{'='*60}")
    print(f"Creating visualizations for {index_name.upper()} WSI index")
    print(f"Using same 80-20 split as training")
    print(f"{'='*60}")
    
    # Paths with index name
    model_path = script_path.with_name(f"gru_model_{index_name}.pth")
    scalers_path = script_path.with_name(f"gru_scalers_{index_name}.pkl")
    feature_cols_path = script_path.with_name(f"gru_feature_cols_{index_name}.pkl")
    split_info_path = script_path.with_name(f"gru_split_info_{index_name}.pkl")
    comparison_plot_path = script_path.with_name(f"gru_state_comparison_{index_name}.pdf")
    timeseries_plot_path = script_path.with_name(f"gru_state_timeseries_{index_name}.pdf")
    
    # Check if model exists
    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}")
        print(f"Skipping {index_name} WSI visualizations.")
        return None
    
    # Load model and split info
    print("Loading trained model and split information...")
    model, scalers, feature_cols, split_info = load_model_and_scalers(
        model_path, scalers_path, feature_cols_path, split_info_path, device
    )
    
    # Get state-wise predictions on test data (last 20%)
    print(f"\nGenerating predictions for all states (test data: last {(1-TRAIN_SPLIT_RATIO)*100:.0f}% of data)...")
    state_results = get_state_predictions(
        df, model, scalers, feature_cols, split_info, target_column,
        sequence_length=SEQUENCE_LENGTH,
        train_split_ratio=TRAIN_SPLIT_RATIO,
        device=device
    )
    
    print(f"Generated predictions for {len(state_results)} states")
    
    # Print split information for verification
    print("\nTest Set Date Ranges (Last 20% of data):")
    for state, result in list(state_results.items())[:5]:  # Show first 5 states
        split_info = result['split_info']
        print(f"  {state}:")
        print(f"    Training ends: {split_info['train_end']}")
        print(f"    Test set: {split_info['test_start']} to {split_info['test_end']}")
        print(f"    Test samples: {split_info['test_samples']} ({split_info['test_samples']/(split_info['train_samples']+split_info['test_samples'])*100:.1f}%)")
    
    # Create comparison plots (scatter plots)
    print("\nCreating state comparison plots (predicted vs actual)...")
    plot_state_comparison(state_results, comparison_plot_path, index_name)
    
    # Create time series plots
    print("\nCreating time series plots...")
    plot_state_timeseries(state_results, timeseries_plot_path, index_name)
    
    # Print summary statistics
    print("\n" + "="*60)
    print(f"VISUALIZATION SUMMARY - {index_name.upper()} WSI")
    print("="*60)
    print(f"Total states visualized: {len(state_results)}")
    
    # Calculate average metrics
    avg_r2 = np.mean([state_results[s]['metrics']['r2'] for s in state_results])
    avg_rmse = np.mean([state_results[s]['metrics']['rmse'] for s in state_results])
    avg_mae = np.mean([state_results[s]['metrics']['mae'] for s in state_results])
    
    print(f"\nAverage Metrics across all states:")
    print(f"  Average R² Score: {avg_r2:.4f}")
    print(f"  Average RMSE: {avg_rmse:.4f}")
    print(f"  Average MAE: {avg_mae:.4f}")
    
    print(f"\nTop 5 States by R² Score:")
    sorted_states = sorted(state_results.items(), 
                          key=lambda x: x[1]['metrics']['r2'], 
                          reverse=True)
    for i, (state, results) in enumerate(sorted_states[:5], 1):
        m = results['metrics']
        print(f"  {i}. {state}: R²={m['r2']:.4f}, RMSE={m['rmse']:.2f}, MAE={m['mae']:.2f}")
    
    print("="*60)
    
    return state_results


def main():
    """Main visualization function - creates visualizations for both WSI indices"""
    # Paths
    script_path = Path(__file__).resolve()
    data_path = script_path.with_name("Final_Statewise_Water_Dataset_preprocessed_WSI.csv")
    
    # Parameters
    SEQUENCE_LENGTH = 12
    TRAIN_SPLIT_RATIO = 0.8  # Must match training split (80% train, 20% test)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nIMPORTANT: Visualizations use ONLY test data (last {(1-TRAIN_SPLIT_RATIO)*100:.0f}% of data)")
    print(f"This ensures visualizations show performance on data NOT seen during training")
    print(f"Using same 80-20 split as training\n")
    
    # WSI indices to visualize
    wsi_indices = {
        'entropy': 'WSI_entropy_0_100',
        'equal': 'WSI_equal_0_100'
    }
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    
    # Create visualizations for both indices
    all_results = {}
    for index_name, target_column in wsi_indices.items():
        results = visualize_single_index(
            df, index_name, target_column, script_path,
            SEQUENCE_LENGTH, TRAIN_SPLIT_RATIO, device
        )
        if results is not None:
            all_results[index_name] = results
    
    # Print comparison summary
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("VISUALIZATION COMPARISON - Both WSI Indices")
        print("="*60)
        print("\nAverage Performance Comparison:")
        print(f"{'Index':<15} {'Avg R²':<12} {'Avg RMSE':<12} {'Avg MAE':<12} {'States':<10}")
        print("-" * 60)
        for index_name, results in all_results.items():
            avg_r2 = np.mean([results[s]['metrics']['r2'] for s in results])
            avg_rmse = np.mean([results[s]['metrics']['rmse'] for s in results])
            avg_mae = np.mean([results[s]['metrics']['mae'] for s in results])
            num_states = len(results)
            print(f"{index_name.upper():<15} {avg_r2:<12.4f} {avg_rmse:<12.4f} "
                  f"{avg_mae:<12.4f} {num_states:<10}")
        print("="*60)
        
        print("\nGenerated Files:")
        for index_name in all_results.keys():
            print(f"\n{index_name.upper()} WSI:")
            print(f"  - gru_state_comparison_{index_name}.pdf (scatter plots)")
            print(f"  - gru_state_timeseries_{index_name}.pdf (time series plots)")


if __name__ == "__main__":
    main()

