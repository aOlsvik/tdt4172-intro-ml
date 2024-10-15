import numpy as np

class LinearRegression():
    
    def __init__(self, learning_rate=0.01, epochs=1000):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = [0, 0]
        self.losses = []


    def compute_gradient(self, X, y):
        m = len(X)
        grad_a = 1/m * np.sum((self.predict(X) - y) * X)
        grad_b = 1/m * np.sum(self.predict(X) - y)
        return grad_a, grad_b
    
    
    def update(self, grad_a, grad_b):
        self.theta[0] = self.theta[0] - self.learning_rate * grad_a
        self.theta[1] = self.theta[1] - self.learning_rate * grad_b
        return
    
    
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        np.random.seed(42)
        self.theta = np.random.randn(2, 1)

        for _ in range(self.epochs):
            mse = np.mean((self.predict(X) - y) ** 2)
            self.losses.append(mse)
            grad_a, grad_b = self.compute_gradient(X, y)
            self.update(grad_a, grad_b)

        return
        
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        return X * self.theta[0] + self.theta[1]
       
