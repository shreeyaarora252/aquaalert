"""
Comprehensive Validation Script for All Forecasting Models
Validates RNN, LSTM, and GRU models on all WSI indices
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
from rnn_model import RNNForecaster, LSTMForecaster, GRUForecaster
from train_rnn import WaterDataset, create_sequences, get_model


def load_model_and_artifacts(model_type, index_name):
    """Load trained model and preprocessing artifacts"""
    print(f"\n  Loading {model_type.upper()} model for {index_name.upper()}...")
    
    # Load scalers
    with open(f'{model_type}_scalers_{index_name}.pkl', 'rb') as f:
        scalers = pickle.load(f)
    
    # Load feature columns
    with open(f'{model_type}_feature_cols_{index_name}.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    
    # Load split info
    with open(f'{model_type}_split_info_{index_name}.pkl', 'rb') as f:
        split_info = pickle.load(f)
    
    # Initialize and load model
    INPUT_SIZE = len(feature_cols)
    model = get_model(model_type, INPUT_SIZE, hidden_size=64, num_layers=2, dropout=0.2)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(f'{model_type}_model_{index_name}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, scalers, feature_cols, split_info, device


def prepare_validation_data(df, target_column, scalers, feature_cols, sequence_length, train_split_ratio=0.8):
    """Prepare validation data using saved scalers"""
    df_sorted = df.sort_values(['year', 'month']).reset_index(drop=True)
    data = df_sorted[feature_cols].values
    target_col_idx = feature_cols.index(target_column)
    
    # Scale data
    scaled_data = np.zeros_like(data)
    for i, col in enumerate(feature_cols):
        scaled_data[:, i] = scalers[col].transform(data[:, i].reshape(-1, 1)).flatten()
    
    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length, target_col_idx)
    
    # Split temporally
    split_idx = int(len(X) * train_split_ratio)
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    
    return X_val, y_val, scalers[target_column]


def evaluate_model(model, X_val, y_val, target_scaler, device):
    """Evaluate model and return metrics"""
    val_dataset = WaterDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.numpy())
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    # Inverse transform
    predictions_original = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals_original = target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actuals_original, predictions_original))
    mae = mean_absolute_error(actuals_original, predictions_original)
    r2 = r2_score(actuals_original, predictions_original)
    mape = np.mean(np.abs((actuals_original - predictions_original) / (actuals_original + 1e-10))) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }


def main():
    """Main validation function - validates all 12 models"""
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL VALIDATION - ALL MODELS")
    print("="*70)
    
    # Load data
    DATA_FILE = 'Final_Statewise_Water_Dataset_preprocessed_WSI_v3.csv'
    print(f"\nLoading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"✓ Dataset loaded: {len(df)} rows\n")
    
    # Define configurations
    model_types = ['rnn', 'lstm', 'gru']
    indices = [
        ('WSI_equal_0_100', 'equal'),
        ('WSI_entropy_0_100', 'entropy'),
        ('WSI_pca_0_100', 'pca'),
        ('WSI_hybrid_0_100', 'hybrid')
    ]
    
    print(f"Validating {len(model_types)} model types × {len(indices)} indices = {len(model_types)*len(indices)} models")
    print("="*70)
    
    all_results = []
    
    # Validate each model
    for model_type in model_types:
        print(f"\n{model_type.upper()} Models:")
        print("-" * 70)
        
        for target_column, index_name in indices:
            # Load and evaluate
            model, scalers, feature_cols, split_info, device = load_model_and_artifacts(model_type, index_name)
            X_val, y_val, target_scaler = prepare_validation_data(
                df, target_column, scalers, feature_cols, split_info['sequence_length']
            )
            
            metrics = evaluate_model(model, X_val, y_val, target_scaler, device)
            
            # Store results
            all_results.append({
                'Model': model_type.upper(),
                'Index': index_name.upper(),
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R2': metrics['R2'],
                'MAPE': metrics['MAPE']
            })
            
            print(f"    {index_name.upper():<10} | RMSE={metrics['RMSE']:6.2f} | MAE={metrics['MAE']:6.2f} | R²={metrics['R2']:6.4f} | MAPE={metrics['MAPE']:5.1f}%")
    
    # Create comprehensive results DataFrame
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('all_models_validation_summary.csv', index=False)
    
    # Find best models for each index
    print("\n" + "="*70)
    print("BEST MODELS BY INDEX (Based on R² Score)")
    print("="*70)
    
    for _, index_name in indices:
        index_results = results_df[results_df['Index'] == index_name.upper()]
        best = index_results.loc[index_results['R2'].idxmax()]
        
        print(f"\n{index_name.upper()} Index:")
        print(f"  🏆 Best Model: {best['Model']}")
        print(f"     RMSE: {best['RMSE']:.2f} | MAE: {best['MAE']:.2f} | R²: {best['R2']:.4f} | MAPE: {best['MAPE']:.1f}%")
    
    # Find overall best model
    print("\n" + "="*70)
    print("OVERALL BEST MODELS")
    print("="*70)
    
    best_r2 = results_df.loc[results_df['R2'].idxmax()]
    best_rmse = results_df.loc[results_df['RMSE'].idxmin()]
    
    print(f"\nHighest R² Score:")
    print(f"  {best_r2['Model']} - {best_r2['Index']}: R²={best_r2['R2']:.4f}, RMSE={best_r2['RMSE']:.2f}")
    
    print(f"\nLowest RMSE:")
    print(f"  {best_rmse['Model']} - {best_rmse['Index']}: RMSE={best_rmse['RMSE']:.2f}, R²={best_rmse['R2']:.4f}")
    
    print("\n" + "="*70)
    print("✓ Validation complete! Results saved to: all_models_validation_summary.csv")
    print("="*70)
    
    return results_df


if __name__ == "__main__":
    main()
