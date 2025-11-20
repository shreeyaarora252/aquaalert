"""
Prediction script for RNN Water Stress Index Forecasting Model
Predicts WSI for the next month for each state
"""
import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
from rnn_model import RNNForecaster


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
    
    # Initialize model
    INPUT_SIZE = len(feature_cols)
    model = RNNForecaster(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, dropout=0.2)
    
    # Load model weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(f'rnn_model_{index_name}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully on {device}")
    
    return model, scalers, feature_cols, split_info, device


def prepare_prediction_data(df, state, feature_cols, scalers, sequence_length):
    """Prepare data for prediction for a specific state"""
    # Filter data for the state
    state_data = df[df['state'] == state].sort_values(['year', 'month']).reset_index(drop=True)
    
    if len(state_data) < sequence_length:
        return None, None
    
    # Get last sequence_length months
    recent_data = state_data.tail(sequence_length)[feature_cols].values
    
    # Scale the data
    scaled_data = np.zeros_like(recent_data)
    for i, col in enumerate(feature_cols):
        scaled_data[:, i] = scalers[col].transform(recent_data[:, i].reshape(-1, 1)).flatten()
    
    # Get last date
    last_date = state_data.iloc[-1]
    
    return scaled_data, last_date


def predict_next_month(model, sequence, device):
    """Predict WSI for next month"""
    # Convert to tensor and add batch dimension
    sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(sequence_tensor)
    
    return prediction.cpu().numpy()[0][0]


def generate_predictions(df, target_column, index_name):
    """Generate predictions for all states"""
    print(f"\n{'#'*60}")
    print(f"# Generating Predictions - {index_name.upper()} WSI")
    print(f"{'#'*60}")
    
    # Load model and artifacts
    model, scalers, feature_cols, split_info, device = load_model_and_artifacts(index_name)
    sequence_length = split_info['sequence_length']
    target_scaler = scalers[target_column]
    
    print(f"\nSequence length: {sequence_length} months")
    print(f"Predicting for {df['state'].nunique()} states")
    
    # Get unique states
    states = sorted(df['state'].unique())
    
    predictions = []
    
    print(f"\nGenerating predictions...")
    for state in states:
        # Prepare data
        sequence, last_date = prepare_prediction_data(
            df, state, feature_cols, scalers, sequence_length
        )
        
        if sequence is None:
            print(f"  ⚠ Skipping {state} (insufficient data)")
            continue
        
        # Make prediction
        predicted_scaled = predict_next_month(model, sequence, device)
        
        # Inverse transform to original scale
        predicted_wsi = target_scaler.inverse_transform([[predicted_scaled]])[0][0]
        
        # Calculate next month
        next_month = int(last_date['month']) + 1
        next_year = int(last_date['year'])
        if next_month > 12:
            next_month = 1
            next_year += 1
        
        # Current WSI for reference
        current_wsi = last_date[target_column]
        
        predictions.append({
            'state': state,
            'last_data_date': f"{int(last_date['year'])}-{int(last_date['month']):02d}",
            'prediction_date': f"{next_year}-{next_month:02d}",
            'current_wsi': current_wsi,
            'predicted_wsi': predicted_wsi,
            'change': predicted_wsi - current_wsi,
            'percent_change': ((predicted_wsi - current_wsi) / (current_wsi + 1e-10)) * 100
        })
        
        print(f"  ✓ {state:25s} | Current: {current_wsi:6.2f} | Predicted: {predicted_wsi:6.2f} | Change: {predicted_wsi - current_wsi:+6.2f}")
    
    # Create DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    # Save predictions
    filename = f'rnn_predictions_{index_name}.csv'
    predictions_df.to_csv(filename, index=False)
    print(f"\n✓ Predictions saved: {filename}")
    
    return predictions_df


def display_summary(predictions_df, index_name):
    """Display summary statistics"""
    print(f"\n{'='*60}")
    print(f"PREDICTION SUMMARY - {index_name.upper()} WSI")
    print(f"{'='*60}")
    
    print(f"\nTotal predictions: {len(predictions_df)}")
    print(f"\nPredicted WSI Statistics:")
    print(f"  Mean:   {predictions_df['predicted_wsi'].mean():.2f}")
    print(f"  Median: {predictions_df['predicted_wsi'].median():.2f}")
    print(f"  Min:    {predictions_df['predicted_wsi'].min():.2f}")
    print(f"  Max:    {predictions_df['predicted_wsi'].max():.2f}")
    print(f"  Std:    {predictions_df['predicted_wsi'].std():.2f}")
    
    print(f"\nExpected Changes:")
    print(f"  Mean change:        {predictions_df['change'].mean():+.2f}")
    print(f"  Mean % change:      {predictions_df['percent_change'].mean():+.2f}%")
    print(f"  States improving:   {len(predictions_df[predictions_df['change'] < 0])} ({len(predictions_df[predictions_df['change'] < 0])/len(predictions_df)*100:.1f}%)")
    print(f"  States worsening:   {len(predictions_df[predictions_df['change'] > 0])} ({len(predictions_df[predictions_df['change'] > 0])/len(predictions_df)*100:.1f}%)")
    
    print(f"\nTop 5 States with Largest Improvement (decrease in WSI):")
    top_improving = predictions_df.nsmallest(5, 'change')[['state', 'current_wsi', 'predicted_wsi', 'change']]
    for idx, row in top_improving.iterrows():
        print(f"  {row['state']:25s}: {row['current_wsi']:6.2f} → {row['predicted_wsi']:6.2f} ({row['change']:+6.2f})")
    
    print(f"\nTop 5 States with Largest Deterioration (increase in WSI):")
    top_worsening = predictions_df.nlargest(5, 'change')[['state', 'current_wsi', 'predicted_wsi', 'change']]
    for idx, row in top_worsening.iterrows():
        print(f"  {row['state']:25s}: {row['current_wsi']:6.2f} → {row['predicted_wsi']:6.2f} ({row['change']:+6.2f})")


def main():
    """Main prediction function"""
    print("\n" + "="*60)
    print("RNN WSI PREDICTION - NEXT MONTH FORECAST")
    print("="*60)
    
    # Load data
    DATA_FILE = 'Final_Statewise_Water_Dataset_preprocessed_WSI.csv'
    print(f"\nLoading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"✓ Dataset loaded: {len(df)} rows, {df['state'].nunique()} states")
    
    # Generate predictions for entropy index
    predictions_entropy = generate_predictions(df, 'WSI_entropy_0_100', 'entropy')
    display_summary(predictions_entropy, 'entropy')
    
    # Generate predictions for equal index
    predictions_equal = generate_predictions(df, 'WSI_equal_0_100', 'equal')
    display_summary(predictions_equal, 'equal')
    
    print("\n" + "="*60)
    print("✓ All predictions completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
