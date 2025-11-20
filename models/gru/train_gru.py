"""
Training script for GRU Water Stress Index Forecasting Model
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import pickle
from gru_model import GRUForecaster, GRUTrainer


class WaterDataset(Dataset):
    """
    Dataset class for water stress index time series data
    """
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
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, target_col_idx])
    
    return np.array(X), np.array(y).reshape(-1, 1)


def prepare_data(df, target_column='WSI_entropy_0_100', sequence_length=12, train_split_ratio=0.8):
    """
    Prepare data for training with temporal 80-20 split
    
    Args:
        df: DataFrame with time series data
        target_column: Column name to predict
        sequence_length: Length of input sequence (default 12 months)
        train_split_ratio: Ratio of data to use for training (default 0.8 = 80%)
        
    Returns:
        X_train, y_train, X_val, y_val, scalers, feature_cols, split_info
    """
    # Feature columns (excluding non-numeric and target)
    exclude_cols = ['year', 'month', 'state', target_column]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Sort by state, year, month (temporal order)
    df = df.sort_values(['state', 'year', 'month']).reset_index(drop=True)
    
    # Process each state separately
    all_X_train, all_y_train = [], []
    all_X_val, all_y_val = [], []
    
    scalers = {}
    split_info = {}  # Store split information for each state
    
    test_split_ratio = 1.0 - train_split_ratio
    print(f"Using temporal split: {train_split_ratio*100:.0f}% for training, {test_split_ratio*100:.0f}% for testing")
    print(f"Split is done chronologically per state (first {train_split_ratio*100:.0f}% for training, last {test_split_ratio*100:.0f}% for testing)")
    
    for state in df['state'].unique():
        state_df = df[df['state'] == state].copy()
        
        if len(state_df) < sequence_length + 1:
            continue
        
        # TEMPORAL 80-20 SPLIT: Split chronologically (first 80% for training, last 20% for testing)
        split_idx = int(len(state_df) * train_split_ratio)
        
        # Need at least sequence_length + 1 samples in training data
        if split_idx < sequence_length + 1:
            print(f"Warning: {state} has insufficient data for split ({len(state_df)} samples), skipping...")
            continue
        
        train_df = state_df.iloc[:split_idx].copy()
        val_df = state_df.iloc[split_idx:].copy()
        
        # Need at least some validation data
        if len(val_df) == 0:
            print(f"Warning: {state} has no validation data after split, skipping...")
            continue
        
        # Store split information
        split_info[state] = {
            'train_start': (train_df.iloc[0]['year'], train_df.iloc[0]['month']),
            'train_end': (train_df.iloc[-1]['year'], train_df.iloc[-1]['month']),
            'test_start': (val_df.iloc[0]['year'], val_df.iloc[0]['month']),
            'test_end': (val_df.iloc[-1]['year'], val_df.iloc[-1]['month']),
            'train_samples': len(train_df),
            'test_samples': len(val_df)
        }
        
        # Fit scaler ONLY on training data to prevent data leakage
        train_features = train_df[feature_cols].values
        train_target = train_df[target_column].values.reshape(-1, 1)
        train_combined = np.hstack([train_features, train_target])
        
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_combined)
        scalers[state] = scaler
        
        # Get target column index (last column)
        target_col_idx = train_scaled.shape[1] - 1
        
        # Create sequences from TRAINING data only
        X_train_state, y_train_state = create_sequences(train_scaled, sequence_length, target_col_idx)
        
        if len(X_train_state) > 0:
            all_X_train.append(X_train_state)
            all_y_train.append(y_train_state)
        
        # For validation, we need to use the last part of training data to create sequences
        # that predict validation data (this is correct - we use training data as context)
        if len(val_df) > 0 and len(train_df) >= sequence_length:
            # Use last sequence_length samples from training as context for first validation prediction
            val_features = val_df[feature_cols].values
            val_target = val_df[target_column].values.reshape(-1, 1)
            val_combined = np.hstack([val_features, val_target])
            
            # Transform validation data using the scaler fitted on training data
            val_scaled = scaler.transform(val_combined)
            
            # Combine last part of training with validation for sequence creation
            combined_for_val = np.vstack([train_scaled[-sequence_length:], val_scaled])
            
            # Create sequences that predict validation data
            X_val_state, y_val_state = create_sequences(combined_for_val, sequence_length, target_col_idx)
            
            if len(X_val_state) > 0:
                all_X_val.append(X_val_state)
                all_y_val.append(y_val_state)
    
    # Concatenate all states
    if len(all_X_train) == 0:
        raise ValueError("No valid training sequences created. Check data and sequence_length.")
    
    X_train = np.vstack(all_X_train)
    y_train = np.vstack(all_y_train)
    
    if len(all_X_val) > 0:
        X_val = np.vstack(all_X_val)
        y_val = np.vstack(all_y_val)
    else:
        print("Warning: No validation sequences created. Using empty validation set.")
        X_val = np.array([]).reshape(0, sequence_length, X_train.shape[2])
        y_val = np.array([]).reshape(0, 1)
    
    print(f"\nTraining samples: {len(X_train)} ({train_split_ratio*100:.0f}% of data)")
    print(f"Validation samples: {len(X_val)} ({test_split_ratio*100:.0f}% of data)")
    
    return X_train, y_train, X_val, y_val, scalers, feature_cols, split_info


def train_single_model(df, target_column, index_name, script_path, 
                       SEQUENCE_LENGTH=12, HIDDEN_SIZE=64, NUM_LAYERS=2, 
                       DROPOUT=0.2, BATCH_SIZE=32, EPOCHS=100, LEARNING_RATE=0.001,
                       TRAIN_SPLIT_RATIO=0.8):
    """
    Train a single GRU model for a specific WSI index
    
    Args:
        df: DataFrame with data
        target_column: Target column name
        index_name: Name of the index ('entropy' or 'equal')
        script_path: Path object for saving files
        TRAIN_SPLIT_RATIO: Ratio of data for training (default 0.8 = 80%)
        Other args: Model hyperparameters
    """
    print(f"\n{'='*60}")
    print(f"Training GRU model for {index_name.upper()} WSI index")
    print(f"Target column: {target_column}")
    print(f"Temporal split: {TRAIN_SPLIT_RATIO*100:.0f}% training, {(1-TRAIN_SPLIT_RATIO)*100:.0f}% testing")
    print(f"{'='*60}")
    
    # Paths with index name
    model_path = script_path.with_name(f"gru_model_{index_name}.pth")
    scalers_path = script_path.with_name(f"gru_scalers_{index_name}.pkl")
    feature_cols_path = script_path.with_name(f"gru_feature_cols_{index_name}.pkl")
    split_info_path = script_path.with_name(f"gru_split_info_{index_name}.pkl")
    
    # Prepare data with temporal 80-20 split
    print("\nPreparing data with temporal 80-20 split...")
    X_train, y_train, X_val, y_val, scalers, feature_cols, split_info = prepare_data(
        df, target_column=target_column, sequence_length=SEQUENCE_LENGTH,
        train_split_ratio=TRAIN_SPLIT_RATIO
    )
    
    print(f"Input features: {X_train.shape[2]}")
    
    # Save split information
    with open(split_info_path, 'wb') as f:
        pickle.dump(split_info, f)
    print(f"Split information saved to {split_info_path}")
    
    # Create datasets
    train_dataset = WaterDataset(X_train, y_train)
    val_dataset = WaterDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    print("\nCreating GRU model...")
    model = GRUForecaster(
        input_size=X_train.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_size=1
    )
    
    # Create trainer
    trainer = GRUTrainer(model, learning_rate=LEARNING_RATE)
    
    # Train model
    print("\nTraining model...")
    history = trainer.train(train_loader, val_loader, epochs=EPOCHS, verbose=True)
    
    # Save model
    trainer.save_model(str(model_path))
    
    # Save scalers and feature columns
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"Scalers saved to {scalers_path}")
    
    with open(feature_cols_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"Feature columns saved to {feature_cols_path}")
    
    # Print split summary
    print("\nSplit Summary (first 3 states):")
    for i, (state, info) in enumerate(list(split_info.items())[:3]):
        print(f"  {state}:")
        print(f"    Training: {info['train_start'][0]}-{info['train_start'][1]:02d} to {info['train_end'][0]}-{info['train_end'][1]:02d} ({info['train_samples']} samples)")
        print(f"    Testing:  {info['test_start'][0]}-{info['test_start'][1]:02d} to {info['test_end'][0]}-{info['test_end'][1]:02d} ({info['test_samples']} samples)")
    
    # Print final metrics
    print("\n" + "="*60)
    print(f"Training completed for {index_name.upper()} WSI!")
    print(f"Final training loss: {history['train_losses'][-1]:.6f}")
    print(f"Final validation loss: {history['val_losses'][-1]:.6f}")
    print(f"Best validation loss: {min(history['val_losses']):.6f}")
    print("="*60)
    
    return history


def main():
    """Main training function - trains both WSI indices separately"""
    # Paths
    script_path = Path(__file__).resolve()
    data_path = script_path.with_name("Final_Statewise_Water_Dataset_preprocessed_WSI.csv")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    
    # Parameters
    SEQUENCE_LENGTH = 12  # 12 months of history
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.2
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    TRAIN_SPLIT_RATIO = 0.8  # 80% for training, 20% for testing
    
    print(f"\nIMPORTANT: Using temporal 80-20 split")
    print(f"  - Training data: First {TRAIN_SPLIT_RATIO*100:.0f}% of data (chronologically)")
    print(f"  - Test data: Last {(1-TRAIN_SPLIT_RATIO)*100:.0f}% of data (chronologically)")
    print(f"  - Split is done per state to maintain temporal order")
    print(f"  - This ensures no data leakage between train and test sets\n")
    
    # WSI indices to train
    wsi_indices = {
        'entropy': 'WSI_entropy_0_100',
        'equal': 'WSI_equal_0_100'
    }
    
    # Train models for both indices
    histories = {}
    for index_name, target_column in wsi_indices.items():
        history = train_single_model(
            df, target_column, index_name, script_path,
            SEQUENCE_LENGTH, HIDDEN_SIZE, NUM_LAYERS,
            DROPOUT, BATCH_SIZE, EPOCHS, LEARNING_RATE,
            TRAIN_SPLIT_RATIO
        )
        histories[index_name] = history
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY - Both WSI Indices")
    print("="*60)
    for index_name, history in histories.items():
        print(f"\n{index_name.upper()} WSI:")
        print(f"  Best validation loss: {min(history['val_losses']):.6f}")
        print(f"  Final validation loss: {history['val_losses'][-1]:.6f}")
    print("="*60)


if __name__ == "__main__":
    main()

