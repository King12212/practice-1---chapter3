import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class LogisticRegressionConfig:
    learning_rate: float = 1e-3
    max_iterations: int = 50000
    tolerance: float = 1e-15
    epsilon: float = 1e-20

class LogisticRegression:
    def __init__(self, config: Optional[LogisticRegressionConfig] = None):
        self.config = config or LogisticRegressionConfig()
        self.params = None
        self.loss_history = []
    
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Apply sigmoid function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Prevent overflow
    
    def _calculate_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate binary cross-entropy loss."""
        m = len(y)
        predictions = self._sigmoid(X @ self.params)
        epsilon = self.config.epsilon
        return float(-(1/m) * (y.T @ np.log(predictions + epsilon) + 
                              (1 - y).T @ np.log(1 - predictions + epsilon)))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """Train the model using gradient descent."""
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        # Add bias term if not present
        if not np.allclose(X[:, 0], 1):
            X = np.c_[np.ones(len(X)), X]
            
        m = len(y)
        self.params = np.zeros((X.shape[1], 1))
        self.loss_history = []
        prev_loss = float('inf')

        for i in range(self.config.max_iterations):
            # Forward pass
            predictions = self._sigmoid(X @ self.params)
            
            # Compute gradients
            gradients = (1/m) * (X.T @ (predictions - y))
            
            # Update parameters
            self.params -= self.config.learning_rate * gradients
            
            # Calculate and store loss
            current_loss = self._calculate_loss(X, y)
            self.loss_history.append(current_loss)
            
            # Check convergence
            if abs(prev_loss - current_loss) < self.config.tolerance:
                print(f"Converged at iteration {i+1}")
                break
                
            prev_loss = current_loss
            
        return self.params, self.loss_history
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Make predictions using trained model."""
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        # Add bias term if not present
        if not np.allclose(X[:, 0], 1):
            X = np.c_[np.ones(len(X)), X]
            
        probabilities = self._sigmoid(X @ self.params)
        return (probabilities >= threshold).astype(int)

class DataVisualizer:
    @staticmethod
    def plot_loss_history(loss_history: List[float], figsize: Tuple[int, int] = (10, 6)):
        """Plot training loss history."""
        plt.figure(figsize=figsize)
        plt.plot(loss_history, color='blue', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_predictions(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, 
                        figsize: Tuple[int, int] = (10, 6)):
        """Plot actual vs predicted values."""
        plt.figure(figsize=figsize)
        plt.scatter(X[:, 1], y, color='blue', label='Actual Data', alpha=0.6)
        plt.scatter(X[:, 1], y_pred, color='red', marker='x', 
                   label='Predicted Data', alpha=0.6)
        plt.xlabel('Grain Size (mm)')
        plt.ylabel('Spider Presence')
        plt.title('Actual vs Predicted Spider Presence')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Data preparation
    grain_sizes = np.array([0.245, 0.247, 0.285, 0.299, 0.327, 0.347, 0.356, 0.36, 0.363, 0.364,
                           0.398, 0.4, 0.409, 0.421, 0.432, 0.473, 0.509, 0.529, 0.561, 0.569,
                           0.594, 0.638, 0.656, 0.816, 0.853, 0.938, 1.036, 1.045])
    spider_presence = np.array([0, 0, 1, 1, 1, 1, 0, 1, 0, 1,
                               0, 1, 0, 1, 0, 1, 1, 1, 0, 0,
                               1, 1, 1, 1, 1, 1, 1, 1]).reshape(-1, 1)

    # Initialize and train model
    config = LogisticRegressionConfig(learning_rate=1e-3, max_iterations=50000, tolerance=1e-15)
    model = LogisticRegression(config)
    X = np.c_[np.ones(len(grain_sizes)), grain_sizes]
    
    # Train model
    params, loss_history = model.fit(X, spider_presence)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Visualize results
    visualizer = DataVisualizer()
    visualizer.plot_loss_history(loss_history)
    visualizer.plot_predictions(X, spider_presence, predictions)
    
    # Calculate and display accuracy
    accuracy = np.mean(predictions == spider_presence) * 100
    print(f"Training Accuracy: {accuracy:.2f}%")