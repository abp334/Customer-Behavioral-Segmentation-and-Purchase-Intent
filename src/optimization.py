"""
src/optimization.py
===================
First-principles implementation of Loss Functions and Gradient Descent algorithms:
  - Batch Gradient Descent (BGD)
  - Stochastic Gradient Descent (SGD)
  - Mini-Batch Gradient Descent (MBGD)
Cost-sensitive threshold tuning and expected value calculations.
Directly aligns with Syllabus Unit 7 (Optimization) and Unit 8 (Supervised ML).
"""

import numpy as np
import pandas as pd


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid activation function."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy_loss(y_true: np.ndarray, y_pred_prob: np.ndarray, eps: float = 1e-15) -> float:
    """
    Binary Cross-Entropy (Log-Loss):
    J(theta) = -(1/n) * sum[ y * log(p) + (1 - y) * log(1 - p) ]
    """
    p = np.clip(y_pred_prob, eps, 1.0 - eps)
    loss = -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    return float(loss)


def brier_score_loss(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """Mean squared error of probabilistic predictions: BS = (1/n) * sum (p_i - y_i)^2."""
    return float(np.mean((y_pred_prob - y_true) ** 2))


class LogisticRegressionFromScratch:
    """
    Logistic Regression classifier trained from first principles using:
      - 'batch': Batch Gradient Descent (full dataset gradient)
      - 'sgd': Stochastic Gradient Descent (single sample per step)
      - 'mini_batch': Mini-Batch Gradient Descent (vectorized batches)
    """

    def __init__(self,
                 learning_rate: float = 0.05,
                 n_epochs: int = 40,
                 batch_size: int = 256,
                 method: str = "mini_batch",
                 random_state: int = 42):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.method = method
        self.random_state = random_state
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionFromScratch":
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        # Xavier / He style small random initialization
        self.weights = rng.normal(0, 0.01, size=n_features)
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.n_epochs):
            if self.method == "batch":
                # Full-batch gradient step
                linear_pred = X @ self.weights + self.bias
                y_pred = sigmoid(linear_pred)
                
                # Gradient: dJ/dw = (1/n) * X^T (y_pred - y)
                dw = (1.0 / n_samples) * (X.T @ (y_pred - y))
                db = (1.0 / n_samples) * np.sum(y_pred - y)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            elif self.method == "sgd":
                # Shuffle every epoch
                indices = rng.permutation(n_samples)
                for idx in indices:
                    xi = X[idx:idx + 1]
                    yi = y[idx:idx + 1]
                    pred = sigmoid(xi @ self.weights + self.bias)
                    dw = xi.T @ (pred - yi)
                    db = pred - yi

                    # Decaying learning rate for SGD convergence
                    lr = self.learning_rate / (1.0 + 0.0001 * (epoch * n_samples + idx))
                    self.weights -= lr * dw.flatten()
                    self.bias -= lr * float(db)

            elif self.method == "mini_batch":
                # Shuffle each epoch
                indices = rng.permutation(n_samples)
                for start in range(0, n_samples, self.batch_size):
                    batch_idx = indices[start:start + self.batch_size]
                    X_b = X[batch_idx]
                    y_b = y[batch_idx]
                    m = len(y_b)

                    pred = sigmoid(X_b @ self.weights + self.bias)
                    dw = (1.0 / m) * (X_b.T @ (pred - y_b))
                    db = (1.0 / m) * np.sum(pred - y_b)

                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db

            # Compute and record loss at the end of each epoch
            current_preds = sigmoid(X @ self.weights + self.bias)
            current_loss = binary_cross_entropy_loss(y, current_preds)
            self.loss_history.append(current_loss)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        linear_pred = X @ self.weights + self.bias
        return sigmoid(linear_pred)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def compute_expected_profit_curve(y_true: np.ndarray,
                                 y_prob: np.ndarray,
                                 revenue_per_buyer: float = 50.0,
                                 cost_per_offer: float = 2.0,
                                 n_thresholds: int = 100) -> pd.DataFrame:
    """
    Calculate the business expected net profit for varying classification thresholds:
      Profit(tau) = TP(tau) * (Revenue - Cost) - FP(tau) * Cost
    """
    thresholds = np.linspace(0.01, 0.90, n_thresholds)
    records = []

    for tau in thresholds:
        y_pred = (y_prob >= tau).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        targeted_users = tp + fp
        # Net Profit: Margin on true buyers minus marketing cost on all targeted users
        profit = (tp * revenue_per_buyer) - (targeted_users * cost_per_offer)
        precision = tp / targeted_users if targeted_users > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        records.append({
            "threshold": round(float(tau), 4),
            "targeted_users": int(targeted_users),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "expected_net_profit": round(float(profit), 2)
        })

    return pd.DataFrame(records)
