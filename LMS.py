import numpy as np


class Lms:
    """ LMS class that implements Widrow-Hoff algorithm
    """
    def __init__(self, eta, n_iter, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.weights = None

    def fit(self, X, y):
        # initialize weights and bias
        self._initWeights((X.shape[1],1))
        for i in range(self.n_iter):
            yPred = self._calcYPred(X)
            arrError = (y - yPred)
            self.weights += self.eta*X.T.dot(arrError).reshape(self.weights.shape)

        return self
    def _calcYPred(self, X):
        return (np.dot(X, self.weights)).flatten()


    def _initWeights(self, weightsShape):
        randomGenerator = np.random.RandomState(self.random_state)
        self.weights = randomGenerator.normal(loc=0.0, scale=0.01, size=weightsShape).astype('float64')

    def calcSquaredError(self, y, y_pred):
        arrError = y - y_pred
        arrError = (arrError**2).sum() / (2.0)
        return arrError

    def predict(self, X):
        return np.where(self._calcYPred(X) < 0.0, 0, 1)