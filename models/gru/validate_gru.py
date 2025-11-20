"""
Validation script for GRU Water Stress Index Forecasting Model
"""
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import pickle
from gru_model import GRUForecaster, GRUTrainer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


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
        split_info = None  # Will be recreated in evaluate_model
    
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


def evaluate_model(
    df,
    model,
    scalers,
    feature_cols,
    split_info,
    target_column='WSI_entropy_0_100',
    sequence_length=12,
    train_split_ratio=0.8,
    device='cpu'
):
    """
    Evaluate model on test data using the same split as training
    
    Args:
        df: Full DataFrame
        model: Trained GRU model
        scalers: Dictionary of scalers
        feature_cols: List of feature columns
        split_info: Dictionary with split information for each state
        target_column: Target column name
        sequence_length: Sequence length
        train_split_ratio: Ratio used for training (to determine test split)
        device: Device to use
        
    Returns:
        Dictionary with evaluation metrics and predictions
    """
    # Sort data
    df = df.sort_values(['state', 'year', 'month']).reset_index(drop=True)
    
    all_predictions = []
    all_actuals = []
    state_results = []
    
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
        # We need to use some training data to create the first sequence
        if len(train_df) >= sequence_length:
            train_features = train_df[feature_cols].values
            train_target = train_df[target_column].values.reshape(-1, 1)
            train_combined = np.hstack([train_features, train_target])
            train_scaled = scaler.transform(train_combined)
            
            # Combine last part of train with test for sequences
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
        
        predictions_unscaled = np.array(predictions_unscaled)
        actuals_unscaled = np.array(actuals_unscaled)
        
        # Calculate metrics for this state
        mse = mean_squared_error(actuals_unscaled, predictions_unscaled)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals_unscaled, predictions_unscaled)
        r2 = r2_score(actuals_unscaled, predictions_unscaled)
        
        state_results.append({
            'state': state,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'num_samples': len(predictions_unscaled)
        })
        
        all_predictions.extend(predictions_unscaled)
        all_actuals.extend(actuals_unscaled)
    
    # Overall metrics
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
    
    overall_mse = mean_squared_error(all_actuals, all_predictions)
    overall_rmse = np.sqrt(overall_mse)
    overall_mae = mean_absolute_error(all_actuals, all_predictions)
    overall_r2 = r2_score(all_actuals, all_predictions)
    
    return {
        'overall_metrics': {
            'mse': overall_mse,
            'rmse': overall_rmse,
            'mae': overall_mae,
            'r2': overall_r2,
            'num_samples': len(all_predictions)
        },
        'state_metrics': pd.DataFrame(state_results),
        'predictions': all_predictions,
        'actuals': all_actuals
    }


def plot_predictions(actuals, predictions, output_path, index_name=''):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    plt.xlabel('Actual WSI')
    plt.ylabel('Predicted WSI')
    title = f'Actual vs Predicted Water Stress Index'
    if index_name:
        title += f' - {index_name.upper()} WSI'
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


def validate_single_index(df, index_name, target_column, script_path,
                          SEQUENCE_LENGTH=12, TRAIN_SPLIT_RATIO=0.8, device='cpu'):
    """
    Validate a single WSI index model using the same 80-20 split as training
    
    Args:
        df: Full DataFrame
        index_name: Name of the index ('entropy' or 'equal')
        target_column: Target column name
        script_path: Path object for file paths
        TRAIN_SPLIT_RATIO: Ratio used for training (0.8 = 80% train, 20% test)
        Other args: Validation parameters
    """
    print(f"\n{'='*60}")
    print(f"Validating {index_name.upper()} WSI index model")
    print(f"Using same 80-20 split as training")
    print(f"{'='*60}")
    
    # Paths with index name
    model_path = script_path.with_name(f"gru_model_{index_name}.pth")
    scalers_path = script_path.with_name(f"gru_scalers_{index_name}.pkl")
    feature_cols_path = script_path.with_name(f"gru_feature_cols_{index_name}.pkl")
    split_info_path = script_path.with_name(f"gru_split_info_{index_name}.pkl")
    results_path = script_path.with_name(f"gru_validation_results_{index_name}.csv")
    plot_path = script_path.with_name(f"gru_validation_plot_{index_name}.png")
    
    # Check if model exists
    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}")
        print(f"Skipping {index_name} WSI validation.")
        return None
    
    # Load model and split info
    print("Loading trained model and split information...")
    model, scalers, feature_cols, split_info = load_model_and_scalers(
        model_path, scalers_path, feature_cols_path, split_info_path, device
    )
    
    # Evaluate model on test data (last 20%)
    print(f"\nEvaluating model on test data (last {(1-TRAIN_SPLIT_RATIO)*100:.0f}% of data)...")
    results = evaluate_model(
        df,
        model,
        scalers,
        feature_cols,
        split_info,
        target_column=target_column,
        sequence_length=SEQUENCE_LENGTH,
        train_split_ratio=TRAIN_SPLIT_RATIO,
        device=device
    )
    
    # Print results
    print("\n" + "="*60)
    print(f"VALIDATION RESULTS - {index_name.upper()} WSI")
    print("="*60)
    print("\nOverall Metrics:")
    print(f"  Number of test samples: {results['overall_metrics']['num_samples']}")
    print(f"  Mean Squared Error (MSE): {results['overall_metrics']['mse']:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {results['overall_metrics']['rmse']:.4f}")
    print(f"  Mean Absolute Error (MAE): {results['overall_metrics']['mae']:.4f}")
    print(f"  R² Score: {results['overall_metrics']['r2']:.4f}")
    
    # Save state-wise results
    results['state_metrics'].to_csv(results_path, index=False)
    print(f"\nState-wise results saved to {results_path}")
    
    # Display top and bottom performing states
    print("\nTop 5 States by R² Score:")
    print(results['state_metrics'].nlargest(5, 'r2')[['state', 'r2', 'rmse', 'mae']])
    
    print("\nBottom 5 States by R² Score:")
    print(results['state_metrics'].nsmallest(5, 'r2')[['state', 'r2', 'rmse', 'mae']])
    
    # Plot predictions
    if len(results['predictions']) > 0:
        plot_predictions(results['actuals'], results['predictions'], plot_path, index_name)
    
    print("="*60)
    
    return results


def main():
    """Main validation function - validates both WSI indices separately"""
    # Paths
    script_path = Path(__file__).resolve()
    data_path = script_path.with_name("Final_Statewise_Water_Dataset_preprocessed_WSI.csv")
    
    # Parameters
    SEQUENCE_LENGTH = 12
    TRAIN_SPLIT_RATIO = 0.8  # Must match training split (80% train, 20% test)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nIMPORTANT: Validation uses ONLY test data (last {(1-TRAIN_SPLIT_RATIO)*100:.0f}% of data)")
    print(f"This ensures validation is on data NOT seen during training")
    print(f"Using same 80-20 split as training\n")
    
    # WSI indices to validate
    wsi_indices = {
        'entropy': 'WSI_entropy_0_100',
        'equal': 'WSI_equal_0_100'
    }
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    
    # Validate both models
    all_results = {}
    for index_name, target_column in wsi_indices.items():
        results = validate_single_index(
            df, index_name, target_column, script_path,
            SEQUENCE_LENGTH, TRAIN_SPLIT_RATIO, device
        )
        if results is not None:
            all_results[index_name] = results
    
    # Print comparison summary
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY - Both WSI Indices")
        print("="*60)
        print("\nOverall Performance Comparison:")
        print(f"{'Index':<15} {'R² Score':<12} {'RMSE':<12} {'MAE':<12} {'Samples':<10}")
        print("-" * 60)
        for index_name, results in all_results.items():
            metrics = results['overall_metrics']
            print(f"{index_name.upper():<15} {metrics['r2']:<12.4f} {metrics['rmse']:<12.4f} "
                  f"{metrics['mae']:<12.4f} {metrics['num_samples']:<10}")
        
        # Determine which performs better
        if all_results['entropy']['overall_metrics']['r2'] > all_results['equal']['overall_metrics']['r2']:
            print("\n✓ Entropy WSI model performs better (higher R²)")
        else:
            print("\n✓ Equal WSI model performs better (higher R²)")
        
        print("="*60)


if __name__ == "__main__":
    main()

