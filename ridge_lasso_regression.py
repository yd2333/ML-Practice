from cmath import inf
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd
from MyRidgeRegression import MyRidgeRegression
from my_cross_val import my_cross_val

# load dataset
X, y = fetch_california_housing(return_X_y=True)

num_data, num_features = X.shape

# shuffle dataset
np.random.seed(2022)
perm = np.random.permutation(num_data)

X = X.tolist()
y = y.tolist()

X = [X[i] for i in perm]
y = [y[i] for i in perm]

X = np.array(X)
y = np.array(y)

# Split dataset into train and test sets
NUM_TRAIN = int(np.ceil(num_data*0.8))
NUM_TEST = num_data - NUM_TRAIN

X_train = X[:NUM_TRAIN]
X_test = X[NUM_TRAIN:]
y_train = y[:NUM_TRAIN]
y_test = y[NUM_TRAIN:]

lambda_vals = [0.01, 0.1, 1, 10, 100]

best_lambda_la = 0
best_lambda_rr = 0
sm_lambda_rr = float('inf')
sm_lambda_la = float('inf')

model_list = []
table_rr = []
table_la = []

for lambda_val in lambda_vals:
    # instantiate ridge regression object
    rr_model = MyRidgeRegression(lambda_val)
    # call to CV function to compute mse for each fold
    mse_list_rr = my_cross_val(rr_model, "mse", X_train, y_train)
    row = mse_list_rr
    mse_rr = sum(mse_list_rr) / len(mse_list_rr)
    row.append(mse_rr) # mean append to right of row
    variance = sum([((x - mse_rr) ** 2) for x in mse_list_rr]) / len(mse_list_rr)
    res = variance ** 0.5
    row.append(res)
    table_rr.append(row)
    if mse_rr < sm_lambda_rr:
        best_lambda_rr = lambda_val
        sm_lambda_rr = mse_rr
    # print mse from CV
    print("MSE after cross validation for Ridge Regression is: ", mse_list_rr)
    # instantiate lasso object
    la_model = Lasso(lambda_val)
    # call to CV function to compute mse for each fold
    mse_list_la = my_cross_val(la_model, "mse", X_train, y_train)
    mse_la = sum(mse_list_la) / len(mse_list_la)
    row = mse_list_la
    row.append(mse_la) # mean append to right of row
    variance = sum([((x - mse_la) ** 2) for x in mse_list_la]) / len(mse_list_la)
    res = variance ** 0.5
    row.append(res)
    table_la.append(row)
    if mse_la < sm_lambda_la:
        best_lambda_la = lambda_val
        sm_lambda_la = mse_la
    # print mse from CV
    print("MSE after cross validation for Lasso Regression is: ", mse_list_la)
# instantiate ridge regression and lasso objects for best values of lambda
model_rr_best = MyRidgeRegression(best_lambda_rr)
model_la_best = Lasso(best_lambda_la)

# fit models using all training data
model_rr_best.fit(X_train, y_train)
model_la_best.fit(X_train, y_train)

# predict on test data
predict_rr = model_rr_best.predict(X_test)
predict_la = model_la_best.predict(X_test)

# compute mse on test data
loss_entries_rr = map(lambda x,y: (x-y)**2, y_test, predict_rr)
loss_rr = sum(loss_entries_rr)/len(predict_rr)

loss_entries_la = map(lambda x,y: (x-y)**2, y_test, predict_la)
loss_la = sum(loss_entries_la)/len(predict_la)
# print mse on test data
print("best lambda for RR: ", best_lambda_rr)
print("best lambda for La: ", best_lambda_la)
print("MSE from Ridge Regression is: ", loss_rr)
print("MSE from Lasso Regression is: ", loss_la)

# df_rr = pd.DataFrame(table_rr, columns = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "Mean", "SD"],
#  index = lambda_vals)
# df_rr.to_csv("Ridge regression.csv")

# df_la = pd.DataFrame(table_la, columns = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "Mean", "SD"],
#  index = lambda_vals)
# df_la.to_csv("lasso regression.csv")
