import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# Complete Simple Linear Regression
# ---------------------------------
class SimpleLinearRegression:
    """
    Simple Linear Regression Model (SLR):
        y = X @ w
    Supports:
        - Full Gradient Descent (GD)
        - Mini-batch Gradient Descent (MBGD)
        - Stochastic Gradient Descent (SGD)
    """

    def __init__(self, t0=200, t1=100000):
        self.t0 = t0                  # initial learning rate parameter
        self.t1 = t1                  # learning rate schedule denominator
        self.w = None                 # weights
        self.w_all = []               # track all weights during training
        self.err_all = []             # track all losses during training

    # --------------------
    # Prediction
    # --------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for given input X.
        X: shape (n_samples, n_features)
        Returns: y_pred (n_samples,)
        """
        return X @ self.w

    # --------------------
    # Loss (MSE / 2)
    # --------------------
    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute half of sum of squared errors
        """
        err = self.predict(X) - y
        return 0.5 * np.sum(err ** 2)

    # --------------------
    # Root Mean Square Error
    # --------------------
    def rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute RMSE over dataset
        """
        return np.sqrt(2 / X.shape[0] * self.loss(X, y))

    # --------------------
    # Gradient Calculation
    # --------------------
    def calculate_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. weights
        """
        return X.T @ (self.predict(X) - y)

    # --------------------
    # Weight Update
    # --------------------
    def update_weights(self, grad: np.ndarray, lr: float) -> np.ndarray:
        """
        Update weights using gradient and learning rate
        """
        self.w = self.w - lr * grad
        return self.w

    # --------------------
    # Learning Rate Schedule
    # --------------------
    def learning_schedule(self, t: int) -> float:
        """
        Learning rate schedule: decays over iterations
        """
        return self.t0 / (t + self.t1)

    # --------------------
    # Full Gradient Descent
    # --------------------
    def gd(self, X: np.ndarray, y: np.ndarray, num_epochs: int, lr: float):
        """
        Gradient Descent (Batch)
        """
        self.w = np.zeros(X.shape[1])
        self.w_all = []
        self.err_all = []

        for epoch in range(num_epochs):
            grad = self.calculate_gradient(X, y)
            self.w_all.append(self.w.copy())
            self.err_all.append(self.loss(X, y))
            self.update_weights(grad, lr)

        return self.w, self.err_all, self.w_all

    # --------------------
    # Mini-batch Gradient Descent
    # --------------------
    def mbgd(self, X: np.ndarray, y: np.ndarray, num_epochs: int, batch_size: int):
        """
        Mini-batch Gradient Descent
        """
        self.w = np.zeros(X.shape[1])
        self.w_all = []
        self.err_all = []

        t = 0  # counter for learning rate schedule
        for epoch in range(num_epochs):
            shuffled_indices = np.random.permutation(X.shape[0])
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]

            for i in range(0, X.shape[0], batch_size):
                t += 1
                xi = X_shuffled[i:i + batch_size]
                yi = y_shuffled[i:i + batch_size]

                grad = 2 / xi.shape[0] * self.calculate_gradient(xi, yi)
                self.w_all.append(self.w.copy())
                self.err_all.append(self.loss(xi, yi))
                self.update_weights(grad, self.learning_schedule(t))

        return self.w, self.err_all, self.w_all

    # --------------------
    # Stochastic Gradient Descent
    # --------------------
    def sgd(self, X: np.ndarray, y: np.ndarray, num_epochs: int):
        """
        Stochastic Gradient Descent
        """
        self.w = np.zeros(X.shape[1])
        self.w_all = []
        self.err_all = []

        t = 0
        for epoch in range(num_epochs):
            for i in range(X.shape[0]):
                t += 1
                random_index = np.random.randint(X.shape[0])
                xi = X[random_index:random_index + 1]
                yi = y[random_index:random_index + 1]

                grad = 2 * self.calculate_gradient(xi, yi)
                self.w_all.append(self.w.copy())
                self.err_all.append(self.loss(xi, yi))
                self.update_weights(grad, self.learning_schedule(t))

        return self.w, self.err_all, self.w_all
