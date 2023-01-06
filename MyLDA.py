from tkinter import W
import numpy as np

class MyLDA():

    def __init__(self, lambda_val):
        self.lambda_val = lambda_val
        self.w = None

    def fit(self, X, y):   
        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((2, 2))
        SB = np.zeros((2, 2))
        for c in np.unique(y):
            mean_c = np.mean(X[y == c], axis=0)
            SW += (X[y == c] - mean_c).T.dot((X[y == c] - mean_c))
            mean_diff = (mean_c - mean_overall).reshape(2, 1)
            SB += X[y == c].shape[0] * (mean_diff).dot(mean_diff.T)
        # Discriminants
        D = np.linalg.inv(SW).dot(SB)
        eigenvalues, eigenvectors = np.linalg.eig(D)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.w = eigenvectors[0]

    def predict(self, X):
        prediction = np.zeros((X.shape[0]))
        projection = np.dot(self.w.T, X.T).reshape((X.shape[0]))
        prediction[projection > self.lambda_val] = 1.
        prediction[projection <= self.lambda_val] = 0.
        return prediction
