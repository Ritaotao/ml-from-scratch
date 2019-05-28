import numpy as np

class LogisticRegression():

    def __init__(self, learning_rate=.1, n_iterations=1000):
        self.learning_rate_ = learning_rate
        self.n_iterations_ = n_iterations
        self.param_ = None

    def _initialize_parameters(self, X):
        n = X.shape[1]
        # initialize with [-1 / sqrt(n_features), 1 / sqrt(n_features)]
        self.param_ = np.random.uniform(-np.sqrt(n), np.sqrt(n), n)
    
    def _sigmoid(self, X):
        Z = np.dot(X, self.param_)
        return 1 / (1 + np.exp(-Z))

    def _propagate(self, X, Y):
        m = X.shape[0]
        epsilon = 1e-8
        A = self._sigmoid(X)
        cost = - np.mean(Y * np.log(A + epsilon) + (1-Y) * np.log(1-A + epsilon))
        grad = 1/m * np.dot(X.T, A - Y)
        return cost, grad
    
    def fit(self, X, Y):
        # add bias term
        X_1 = np.c_[np.ones(X.shape[0]), X]
        # initialize parameters
        self._initialize_parameters(X_1)

        for i in range(self.n_iterations_):
            cost, grad = self._propagate(X_1, Y)
            self.param_ -= self.learning_rate_ * grad
        return

    def predict(self, X, prob=False):
        # add bias term
        X_1 = np.c_[np.ones(X.shape[0]), X]
        A = self._sigmoid(X_1)
        if prob:
            return A
        else:
            return (A >= 0.5).astype(int)