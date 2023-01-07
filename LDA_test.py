import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from my_cross_val import my_cross_val
from MyLDA import MyLDA

# load dataset
data = pd.read_csv('LDA_dataset.csv', header=None).to_numpy()
X = data[:,:-1]
y = data[:,-1]

num_data, num_features = X.shape

plt.scatter(X[:1000, 0], X[:1000, 1])
plt.scatter(X[1000:, 0], X[1000:, 1])
plt.show()

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

# train & test

lambda_vals = np.linspace(-2,2,20).tolist()
model_lists = []
cv_results = []

table = []
for lambda_val in lambda_vals:
    # instantiate LDA object
    model = MyLDA(lambda_val)
    # call to CV function to compute error rates for each fold
    err_rates = my_cross_val(model, 'err_rate', X_train, y_train, k=10)

    row = err_rates
    err = sum(err_rates) / len(err_rates)
    row.append(err) # mean append to right of row
    variance = sum([((x - err) ** 2) for x in err_rates]) / len(err_rates)
    res = variance ** 0.5
    row.append(res)
    table.append(row)


    cv_results.append(err_rates)
    model_lists.append(model)
    # print error rates from CV
    print("when lambda = ", lambda_val, ", Error rate after cross validation for LDA is: ", err_rates)



# instantiaste LDA object for best value of lambda
cv_results = np.array(cv_results)
cv_result_means = np.mean(cv_results, axis=1)
idx_best = np.argmin(cv_result_means)

best_LDA = model_lists[idx_best]
best_lambda = lambda_vals[idx_best]
# fit model using all training data
best_LDA.fit(X_train,y_train)

# predict on test data
prediction = best_LDA.predict(X_test)

# compute error rate on test data
loss_entries = map(lambda x,y: abs(x-y)/2, prediction, y_test)
loss = sum(loss_entries)/len(y_test)

# print error rate on test data
print("Error Rate from LDA is: ", loss)
# -0.002
# Error Rate from LDA is:  0.0125

# df = pd.DataFrame(table, columns = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "Mean", "SD"],
#  index = lambda_vals)
# df.to_csv("LDA.csv")