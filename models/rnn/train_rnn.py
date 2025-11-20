"""
Training script for RNN Water Stress Index Forecasting Model
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import pickle
from rnn_model import RNNForecaster, RNNTrainer


class WaterDataset(Dataset):
    """Dataset class for water stress index time series data"""
    
    def __init__(self, sequences, targets):
        """
        Args:
            sequences: Input sequences (num_samples, sequence_length, num_features)
            targets: Target values (num_samples, 1)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def create_sequences(data, sequence_length, target_col_idx):
    """
    Create sequences for time series forecasting
    
    Args:
        data: 2D numpy array (num_samples, num_features)
        sequence_length: Length of input sequence
        target_col_idx: Index of target column
        
    Returns:
        X: Input sequences (num_samples - sequence_length, sequence_length, num_features)
        y: Target values (num_samples - sequence_length, 1)
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length, target_col_idx])
    
    return np.array(X), np.array(y).reshape(-1, 1)


def prepare_data(df, target_column='WSI_entropy_0_100', sequence_length=12, train_split_ratio=0.8):
    """
    Prepare data for training with temporal 80-20 split
    
    Args:
        df: DataFrame with time series data
        target_column: Column name to predict
        sequence_length: Length of input sequence (default 12 months)
        train_split_ratio: Ratio for train-validation split (default 0.8)
        
    Returns:
        X_train, y_train, X_val, y_val, scalers, feature_cols, split_info
    """
    print(f"\n{'='*60}")
    print(f"Preparing data for {target_column}")
    print(f"{'='*60}")
    
    # Sort by year and month to ensure temporal order
    df_sorted = df.sort_values(['year', 'month']).reset_index(drop=True)
    
    # Select features for the model
    feature_cols = [
        'rainfall', 'soil_moisture', 'groundwater_level', 
        'population', 'population_consumption_per_month', 'LPCD',
        'rainfall_z', 'soil_moisture_z', 'groundwater_z', 'LPCD_z',
        target_column
    ]
    
    print(f"\nFeatures used ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  - {col}")
    
    # Extract features
    data = df_sorted[feature_cols].values
    
    # Get target column index
    target_col_idx = feature_cols.index(target_column)
    
    print(f"\nData shape: {data.shape}")
    print(f"Target column: {target_column} (index: {target_col_idx})")
    
    # Initialize scalers for each feature
    scalers = {}
    scaled_data = np.zeros_like(data)
    
    # Scale each feature independently
    for i, col in enumerate(feature_cols):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data[:, i] = scaler.fit_transform(data[:, i].reshape(-1, 1)).flatten()
        scalers[col] = scaler
    
    print(f"\nScaling completed for {len(scalers)} features")
    
    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length, target_col_idx)
    
    print(f"\nSequence creation:")
    print(f"  - Sequence length: {sequence_length} months")
    print(f"  - Total sequences: {len(X)}")
    print(f"  - Input shape: {X.shape}")
    print(f"  - Target shape: {y.shape}")
    
    # Calculate split point for temporal 80-20 split
    split_idx = int(len(X) * train_split_ratio)
    
    # Split data temporally (first 80% for training, last 20% for validation)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    
    print(f"\nData split (temporal {int(train_split_ratio*100)}-{int((1-train_split_ratio)*100)}):")
    print(f"  - Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  - Validation samples: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    
    # Calculate date ranges for split
    original_dates = df_sorted[['year', 'month']].iloc[sequence_length:].reset_index(drop=True)
    train_start = original_dates.iloc[0]
    train_end = original_dates.iloc[split_idx-1]
    val_start = original_dates.iloc[split_idx]
    val_end = original_dates.iloc[-1]
    
    print(f"\nTemporal coverage:")
    print(f"  - Training: {int(train_start['year'])}-{int(train_start['month']):02d} to {int(train_end['year'])}-{int(train_end['month']):02d}")
    print(f"  - Validation: {int(val_start['year'])}-{int(val_start['month']):02d} to {int(val_end['year'])}-{int(val_end['month']):02d}")
    
    split_info = {
        'total_sequences': len(X),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'sequence_length': sequence_length,
        'train_date_range': f"{int(train_start['year'])}-{int(train_start['month']):02d} to {int(train_end['year'])}-{int(train_end['month']):02d}",
        'val_date_range': f"{int(val_start['year'])}-{int(val_start['month']):02d} to {int(val_end['year'])}-{int(val_end['month']):02d}"
    }
    
    return X_train, y_train, X_val, y_val, scalers, feature_cols, split_info


def train_single_model(df, target_column, index_name, 
                       SEQUENCE_LENGTH=12, HIDDEN_SIZE=64, NUM_LAYERS=2, 
                       DROPOUT=0.2, BATCH_SIZE=32, EPOCHS=100, LEARNING_RATE=0.001,
                       TRAIN_SPLIT_RATIO=0.8):
    """
    Train a single RNN model for a specific WSI index
    
    Args:
        df: DataFrame with data
        target_column: Target column name
        index_name: Name of the index ('entropy' or 'equal')
        Other parameters: Model and training hyperparameters
    """
    print(f"\n{'#'*60}")
    print(f"# Training RNN Model for WSI {index_name.upper()}")
    print(f"{'#'*60}")
    
    # Prepare data
    X_train, y_train, X_val, y_val, scalers, feature_cols, split_info = prepare_data(
        df, target_column, SEQUENCE_LENGTH, TRAIN_SPLIT_RATIO
    )
    
    # Create datasets and dataloaders
    train_dataset = WaterDataset(X_train, y_train)
    val_dataset = WaterDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\nDataLoader configuration:")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    
    # Initialize model
    INPUT_SIZE = X_train.shape[2]  # Number of features
    
    print(f"\nModel configuration:")
    print(f"  - Input size: {INPUT_SIZE} features")
    print(f"  - Hidden size: {HIDDEN_SIZE}")
    print(f"  - Number of layers: {NUM_LAYERS}")
    print(f"  - Dropout: {DROPOUT}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Max epochs: {EPOCHS}")
    
    model = RNNForecaster(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_size=1
    )
    
    # Initialize trainer
    trainer = RNNTrainer(model, learning_rate=LEARNING_RATE)
    
    print(f"\nDevice: {trainer.device}")
    print("\nStarting training...\n")
    
    # Train model
    history = trainer.train(train_loader, val_loader, epochs=EPOCHS, verbose=True)
    
    # Save model and artifacts
    print(f"\nSaving model and artifacts...")
    trainer.save_model(f'rnn_model_{index_name}.pth')
    
    # Save scalers
    with open(f'rnn_scalers_{index_name}.pkl', 'wb') as f:
        pickle.dump(scalers, f)
    print(f"Scalers saved to rnn_scalers_{index_name}.pkl")
    
    # Save feature columns
    with open(f'rnn_feature_cols_{index_name}.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"Feature columns saved to rnn_feature_cols_{index_name}.pkl")
    
    # Save split information
    with open(f'rnn_split_info_{index_name}.pkl', 'wb') as f:
        pickle.dump(split_info, f)
    print(f"Split info saved to rnn_split_info_{index_name}.pkl")
    
    print(f"\n✓ Training completed for {index_name.upper()} index")
    print(f"  Final training loss: {history['train_losses'][-1]:.6f}")
    print(f"  Final validation loss: {history['val_losses'][-1]:.6f}")
    
    return history, model, trainer


def main():
    """Main training function - trains both WSI indices separately"""
    print("\n" + "="*60)
    print("RNN MODEL TRAINING FOR WATER STRESS INDEX PREDICTION")
    print("="*60)
    
    # Load preprocessed data
    DATA_FILE = 'Final_Statewise_Water_Dataset_preprocessed_WSI.csv'
    
    print(f"\nLoading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    
    print(f"Dataset loaded successfully!")
    print(f"  - Total rows: {len(df)}")
    print(f"  - Total columns: {len(df.columns)}")
    print(f"  - Date range: {int(df['year'].min())}-{int(df['month'].min()):02d} to {int(df['year'].max())}-{int(df['month'].max()):02d}")
    print(f"  - Unique states: {df['state'].nunique()}")
    
    # Model hyperparameters
    SEQUENCE_LENGTH = 12  # Use 12 months of history
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.2
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    TRAIN_SPLIT_RATIO = 0.8  # 80% training, 20% validation
    
    print(f"\nHyperparameters:")
    print(f"  - Sequence length: {SEQUENCE_LENGTH} months")
    print(f"  - Hidden size: {HIDDEN_SIZE}")
    print(f"  - Layers: {NUM_LAYERS}")
    print(f"  - Dropout: {DROPOUT}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Train/Val split: {int(TRAIN_SPLIT_RATIO*100)}/{int((1-TRAIN_SPLIT_RATIO)*100)}")
    
    # Train model for entropy-weighted WSI
    history_entropy, model_entropy, trainer_entropy = train_single_model(
        df, 
        target_column='WSI_entropy_0_100',
        index_name='entropy',
        SEQUENCE_LENGTH=SEQUENCE_LENGTH,
        HIDDEN_SIZE=HIDDEN_SIZE,
        NUM_LAYERS=NUM_LAYERS,
        DROPOUT=DROPOUT,
        BATCH_SIZE=BATCH_SIZE,
        EPOCHS=EPOCHS,
        LEARNING_RATE=LEARNING_RATE,
        TRAIN_SPLIT_RATIO=TRAIN_SPLIT_RATIO
    )
    
    # Train model for equal-weighted WSI
    history_equal, model_equal, trainer_equal = train_single_model(
        df, 
        target_column='WSI_equal_0_100',
        index_name='equal',
        SEQUENCE_LENGTH=SEQUENCE_LENGTH,
        HIDDEN_SIZE=HIDDEN_SIZE,
        NUM_LAYERS=NUM_LAYERS,
        DROPOUT=DROPOUT,
        BATCH_SIZE=BATCH_SIZE,
        EPOCHS=EPOCHS,
        LEARNING_RATE=LEARNING_RATE,
        TRAIN_SPLIT_RATIO=TRAIN_SPLIT_RATIO
    )
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"\n✓ Entropy-weighted WSI model:")
    print(f"    Final Training Loss: {history_entropy['train_losses'][-1]:.6f}")
    print(f"    Final Validation Loss: {history_entropy['val_losses'][-1]:.6f}")
    print(f"    Epochs trained: {len(history_entropy['train_losses'])}")
    
    print(f"\n✓ Equal-weighted WSI model:")
    print(f"    Final Training Loss: {history_equal['train_losses'][-1]:.6f}")
    print(f"    Final Validation Loss: {history_equal['val_losses'][-1]:.6f}")
    print(f"    Epochs trained: {len(history_equal['train_losses'])}")
    
    print("\n" + "="*60)
    print("All models trained successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
