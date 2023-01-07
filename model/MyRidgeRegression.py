import numpy as np

class MyRidgeRegression():

    def __init__(self, lambda_val):
        self.lambda_val = lambda_val
        self.w_hat = None
        

    def fit(self, X, y):
        _, dim2 = X.shape
        X_transpose = np.matrix(np.transpose(X))
        y = np.expand_dims(y, axis=1)
        self.w_hat = np.linalg.inv(X_transpose*X + self.lambda_val*np.eye(dim2))*X_transpose*y


    def predict(self, X):
        predictions_rr = []
        for x in X:
            pred_target = np.dot(np.transpose(self.w_hat), x)
            predictions_rr.append(pred_target[0,0])
        return predictions_rr
       

