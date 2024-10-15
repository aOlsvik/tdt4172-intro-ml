import numpy as np

class LogisticRegression():
    
    def __init__(self, learning_rate=0.01, epochs=1000, classify_threshold=0.5):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.classify_threshold = classify_threshold
        
        self.weights, self.bias = None, None
        self.losses, self.accuracies = [], []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y, y_pred):
        m = len(y)
        return -1/m * np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
    
    def compute_gradients(self, X, y, y_pred):
        m = len(y)
        grad_w = 1/m * np.dot((y_pred - y), X)
        grad_b = 1/m * np.sum(y_pred - y)
        return grad_w, grad_b
    
    def update_parameters(self, grad_w, grad_b):
        self.weights = self.weights - self.learning_rate * grad_w
        self.bias = self.bias - self.learning_rate * grad_b
        return
    
    def accuracy(self, y, y_pred):
        return np.mean(y == y_pred)
    
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for _ in range(self.epochs):
            lin_pred = np.dot(X, self.weights) + self.bias
            sigm_pred = self.sigmoid(lin_pred)
            
            grad_w, grad_b = self.compute_gradients(X, y, sigm_pred)
            self.update_parameters(grad_w, grad_b)

            loss = self.compute_loss(y, sigm_pred)
            pred_to_class = [1 if _y > self.classify_threshold else 0 for _y in sigm_pred]
            self.accuracies.append(self.accuracy(y, pred_to_class))
            self.losses.append(loss)
    
    def predict(self, X):
        lin_pred = np.dot(X, self.weights) + self.bias
        sigm_pred = self.sigmoid(lin_pred)
        
        return [1 if y > self.classify_threshold else 0 for y in sigm_pred]
       
    def predict_proba(self, X):
        lin_pred = np.dot(X, self.weights) + self.bias
        sigm_pred = self.sigmoid(lin_pred)
        
        return sigm_pred
