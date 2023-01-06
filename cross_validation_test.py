from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from my_cross_val import my_cross_val

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# load regression dataset
X, y = fetch_california_housing(return_X_y=True)

# instantiate ridge regression object
rr_model = Ridge(alpha=0.01)

# call to CV function
mse_vals = my_cross_val(rr_model, 'mse', X, y, k=10)

print('Ridge regression CV MSE values', mse_vals)


# load classification dataset
X, y = load_breast_cancer(return_X_y=True)

# instantiate logistic regression object
lr_model = LogisticRegression()

# call to CV function
err_rates = my_cross_val(lr_model, 'err_rate', X, y, k=10)

print('Logistic Regression CV error rates', err_rates)
