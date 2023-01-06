from statistics import mode
import numpy as np

def my_cross_val(model, loss_func, X, y, k=10):
    # split the dataset to k folds, list of lists
    # indx = np.floor(np.linspace(0, X.shape[0],k).tolist())
    X_folds = []
    Y_folds = []
    count = len(X) // k ## floor division
    for i in range(k):
        if i == k-1:
            X_folds.append(X[i*count:])
            Y_folds.append(y[i*count:])
        else:
            X_folds.append(X[i*count:((i+1)*count)])
            Y_folds.append(y[i*count:((i+1)*count)])

    # iterately train the model
    loss_list = []
    for j in range(k):
        # concat the dataframe other than the jth
        X_train = np.concatenate(X_folds[0:j]+X_folds[(j+1):])
        Y_train = np.concatenate(Y_folds[0:j]+Y_folds[(j+1):])
        model.fit(X_train,Y_train)
        y_predict = model.predict(X_folds[j])
        
        if loss_func == "mse":
            loss_entries = map(lambda x,y: (x-y)**2, y_predict, Y_folds[j])
            loss = sum(loss_entries)/len(y_predict)
        else:
            # loss_entries = map(lambda x,y: abs(x-y)/2, y_predict, Y_folds[j])
            loss = np.mean( y_predict != Y_folds[j])
            # loss = sum(loss_entries)/len(y_predict)

        loss_list.append(loss)
    return loss_list