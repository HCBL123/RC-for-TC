import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class ReservoirNetwork:
    def __init__(self, input_dim, reservoir_size=400, spectral_radius=0.95, sparsity=0.1, noise=0.001):
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize input weights with better scaling
        self.W_in = torch.randn(reservoir_size, input_dim, device=self.device) / np.sqrt(input_dim)
        
        # Create sparse reservoir matrix
        W = torch.randn(reservoir_size, reservoir_size, device=self.device) / np.sqrt(reservoir_size)
        mask = (torch.rand(reservoir_size, reservoir_size, device=self.device) > sparsity).float()
        self.W = W * mask
        
        # Scale reservoir matrix
        with torch.no_grad():
            eigenvals = torch.linalg.eigvals(self.W.cpu()).abs()
            radius = eigenvals.max().item()
            self.W *= (spectral_radius / radius)
        
        self.state = torch.zeros(reservoir_size, device=self.device)
        
        # Define readout layer for 6 emotion classes
        self.readout = nn.Linear(reservoir_size, 6).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=None)  # Will set class weights during training
        self.optimizer = torch.optim.Adam(self.readout.parameters(), lr=0.001)
        
        self.scaler = StandardScaler()
    
    def _update_state(self, input_data, reset=False):
        """Update reservoir state"""
        if reset:
            self.state = torch.zeros(self.reservoir_size, device=self.device)
        
        # Convert input to tensor if it's not already
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, dtype=torch.float32, device=self.device)
        
        # Simplified state update with leaky integration
        leak_rate = 0.3
        preactivation = torch.matmul(self.W_in, input_data) + torch.matmul(self.W, self.state)
        self.state = (1 - leak_rate) * self.state + leak_rate * torch.tanh(preactivation)
        self.state += self.noise * torch.randn_like(self.state)
        
        return self.state
    
    def compute_class_weights(self, y):
        """Compute balanced class weights"""
        counts = torch.bincount(torch.tensor(y))
        weights = 1.0 / counts.float()
        weights = weights / weights.sum() * len(weights)
        return weights.to(self.device)
    
    def train(self, X, y, batch_size=64, epochs=10):
        """Train the reservoir network with batching and GPU acceleration"""
        # Convert labels to tensor
        y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        
        # Compute class weights
        class_weights = self.compute_class_weights(y)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Create dataset
        n_samples = len(X)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        best_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(epochs):
            self.readout.train()
            total_loss = 0
            
            # Shuffle indices
            indices = torch.randperm(n_samples)
            
            # Process in batches
            for i in tqdm(range(0, n_samples, batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
                batch_indices = indices[i:i+batch_size]
                batch_X = X[batch_indices.cpu().numpy()]
                batch_y = y_tensor[batch_indices]
                
                # Compute reservoir states
                batch_states = []
                for x in batch_X:
                    state = self._update_state(x, reset=True)
                    batch_states.append(state)
                
                # Convert states to tensor
                states = torch.stack(batch_states)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.readout(states)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # Calculate average loss for this epoch
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                torch.save(self.readout.state_dict(), 'models/best_readout.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    # Load best model
                    self.readout.load_state_dict(torch.load('models/best_readout.pt'))
                    break
    
    def predict(self, X, batch_size=64):
        """Predict using the reservoir network with GPU acceleration"""
        self.readout.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_states = []
                
                for x in batch_X:
                    state = self._update_state(x, reset=True)
                    batch_states.append(state)
                
                states = torch.stack(batch_states)
                outputs = self.readout(states)
                pred = outputs.argmax(dim=1)
                predictions.extend(pred.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, X, batch_size=64):
        """Predict class probabilities using the reservoir network"""
        self.readout.eval()
        probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_states = []
                
                for x in batch_X:
                    state = self._update_state(x, reset=True)
                    batch_states.append(state)
                
                states = torch.stack(batch_states)
                outputs = self.readout(states)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)