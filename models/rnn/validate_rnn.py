"""
Validation script for RNN Water Stress Index Forecasting Model
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
from rnn_model import RNNForecaster
from train_rnn import WaterDataset, create_sequences


def load_model_and_artifacts(index_name):
    """Load trained model and preprocessing artifacts"""
    print(f"\nLoading model for {index_name.upper()} index...")
    
    # Load scalers
    with open(f'rnn_scalers_{index_name}.pkl', 'rb') as f:
        scalers = pickle.load(f)
    
    # Load feature columns
    with open(f'rnn_feature_cols_{index_name}.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    
    # Load split info
    with open(f'rnn_split_info_{index_name}.pkl', 'rb') as f:
        split_info = pickle.load(f)
    
    print(f"✓ Artifacts loaded successfully")
    print(f"  - Features: {len(feature_cols)}")
    print(f"  - Scalers: {len(scalers)}")
    print(f"  - Sequence length: {split_info['sequence_length']}")
    
    # Initialize model
    INPUT_SIZE = len(feature_cols)
    model = RNNForecaster(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, dropout=0.2)
    
    # Load model weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(f'rnn_model_{index_name}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on device: {device}")
    
    return model, scalers, feature_cols, split_info, device


def prepare_validation_data(df, target_column, scalers, feature_cols, sequence_length, train_split_ratio=0.8):
    """Prepare validation data using saved scalers"""
    print(f"\nPreparing validation data...")
    
    # Sort by year and month
    df_sorted = df.sort_values(['year', 'month']).reset_index(drop=True)
    
    # Extract features
    data = df_sorted[feature_cols].values
    target_col_idx = feature_cols.index(target_column)
    
    # Scale data using saved scalers
    scaled_data = np.zeros_like(data)
    for i, col in enumerate(feature_cols):
        scaled_data[:, i] = scalers[col].transform(data[:, i].reshape(-1, 1)).flatten()
    
    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length, target_col_idx)
    
    # Split temporally (same as training)
    split_idx = int(len(X) * train_split_ratio)
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    
    # Get dates for validation set
    val_dates = df_sorted[['year', 'month', 'state']].iloc[sequence_length + split_idx:].reset_index(drop=True)
    
    print(f"✓ Data prepared")
    print(f"  - Validation samples: {len(X_val)}")
    
    return X_val, y_val, val_dates, scalers[target_column]


def evaluate_model(model, X_val, y_val, target_scaler, device):
    """Evaluate model on validation set"""
    print(f"\nEvaluating model...")
    
    # Create DataLoader
    val_dataset = WaterDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Get predictions
    predictions = []
    actuals = []
    
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.numpy())
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    # Inverse transform to original scale
    predictions_original = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals_original = target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(actuals_original, predictions_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals_original, predictions_original)
    r2 = r2_score(actuals_original, predictions_original)
    mape = np.mean(np.abs((actuals_original - predictions_original) / (actuals_original + 1e-10))) * 100
    
    print(f"\n{'='*50}")
    print(f"VALIDATION METRICS (Original Scale)")
    print(f"{'='*50}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"{'='*50}")
    
    return predictions_original, actuals_original, {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }


def plot_validation_results(predictions, actuals, val_dates, index_name, metrics):
    """Plot validation results"""
    print(f"\nGenerating validation plots...")
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Actual vs Predicted over time
    ax1 = axes[0]
    x_axis = range(len(predictions))
    ax1.plot(x_axis, actuals, label='Actual', color='blue', linewidth=2, alpha=0.7)
    ax1.plot(x_axis, predictions, label='Predicted', color='red', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('WSI Value', fontsize=12)
    ax1.set_title(f'RNN Model Validation - {index_name.upper()} WSI\nR² = {metrics["R2"]:.4f}, RMSE = {metrics["RMSE"]:.4f}', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    ax2 = axes[1]
    ax2.scatter(actuals, predictions, alpha=0.5, s=50)
    
    # Perfect prediction line
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual WSI', fontsize=12)
    ax2.set_ylabel('Predicted WSI', fontsize=12)
    ax2.set_title(f'Actual vs Predicted Scatter Plot\nMAE = {metrics["MAE"]:.4f}, MAPE = {metrics["MAPE"]:.2f}%', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'rnn_validation_plot_{index_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {filename}")
    plt.close()


def save_validation_results(predictions, actuals, val_dates, index_name, metrics):
    """Save validation results to CSV"""
    print(f"\nSaving validation results...")
    
    results_df = pd.DataFrame({
        'year': val_dates['year'].astype(int),
        'month': val_dates['month'].astype(int),
        'state': val_dates['state'],
        'actual_wsi': actuals,
        'predicted_wsi': predictions,
        'absolute_error': np.abs(actuals - predictions),
        'percentage_error': np.abs((actuals - predictions) / (actuals + 1e-10)) * 100
    })
    
    filename = f'rnn_validation_results_{index_name}.csv'
    results_df.to_csv(filename, index=False)
    print(f"✓ Results saved: {filename}")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_filename = f'rnn_validation_metrics_{index_name}.csv'
    metrics_df.to_csv(metrics_filename, index=False)
    print(f"✓ Metrics saved: {metrics_filename}")
    
    return results_df


def validate_single_model(df, target_column, index_name):
    """Validate a single model"""
    print(f"\n{'#'*60}")
    print(f"# Validating RNN Model - {index_name.upper()} WSI")
    print(f"{'#'*60}")
    
    # Load model and artifacts
    model, scalers, feature_cols, split_info, device = load_model_and_artifacts(index_name)
    
    # Prepare validation data
    X_val, y_val, val_dates, target_scaler = prepare_validation_data(
        df, target_column, scalers, feature_cols, 
        split_info['sequence_length'], train_split_ratio=0.8
    )
    
    # Evaluate model
    predictions, actuals, metrics = evaluate_model(model, X_val, y_val, target_scaler, device)
    
    # Plot results
    plot_validation_results(predictions, actuals, val_dates, index_name, metrics)
    
    # Save results
    results_df = save_validation_results(predictions, actuals, val_dates, index_name, metrics)
    
    print(f"\n✓ Validation completed for {index_name.upper()} index")
    
    return metrics, results_df


def main():
    """Main validation function"""
    print("\n" + "="*60)
    print("RNN MODEL VALIDATION")
    print("="*60)
    
    # Load data
    DATA_FILE = 'Final_Statewise_Water_Dataset_preprocessed_WSI.csv'
    print(f"\nLoading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"✓ Dataset loaded: {len(df)} rows")
    
    # Validate entropy model
    metrics_entropy, results_entropy = validate_single_model(
        df, 'WSI_entropy_0_100', 'entropy'
    )
    
    # Validate equal model
    metrics_equal, results_equal = validate_single_model(
        df, 'WSI_equal_0_100', 'equal'
    )
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    print(f"\nEntropy-weighted WSI:")
    print(f"  RMSE: {metrics_entropy['RMSE']:.4f}")
    print(f"  MAE:  {metrics_entropy['MAE']:.4f}")
    print(f"  R²:   {metrics_entropy['R2']:.4f}")
    print(f"  MAPE: {metrics_entropy['MAPE']:.2f}%")
    
    print(f"\nEqual-weighted WSI:")
    print(f"  RMSE: {metrics_equal['RMSE']:.4f}")
    print(f"  MAE:  {metrics_equal['MAE']:.4f}")
    print(f"  R²:   {metrics_equal['R2']:.4f}")
    print(f"  MAPE: {metrics_equal['MAPE']:.2f}%")
    
    print("\n" + "="*60)
    print("✓ All validations completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
