"""
GRU Model for Water Stress Index Forecasting
"""
import torch
import torch.nn as nn
import numpy as np


class GRUForecaster(nn.Module):
    """
    GRU-based model for time series forecasting of Water Stress Index
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """
        Initialize GRU Forecaster
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in GRU layers
            num_layers: Number of GRU layers
            dropout: Dropout rate
            output_size: Number of output values to predict
        """
        super(GRUForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # GRU forward pass
        out, _ = self.gru(x, h0)
        
        # Take the last output from the sequence
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Fully connected layer
        out = self.fc(out)
        
        return out


class GRUTrainer:
    """
    Trainer class for GRU model
    """
    def __init__(
        self,
        model: GRUForecaster,
        learning_rate: float = 0.001,
        device: str = None
    ):
        """
        Initialize trainer
        
        Args:
            model: GRUForecaster model instance
            learning_rate: Learning rate for optimizer
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # Loss function (Mean Squared Error)
        self.criterion = nn.MSELoss()
        
        # Optimizer (Adam)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self, val_loader):
        """
        Validate the model
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self, train_loader, val_loader, epochs: int = 100, verbose: bool = True):
        """
        Train the model
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")

