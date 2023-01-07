import numpy as np



class GaussianDiscriminant_C1:
    # classifier initialization
    # input:
    #   k: number of classes (2 for this assignment)
    #   d: number of features; feature dimensions (8 for this assignment)
    def __init__(self, k=2, d=8):
        self.m = np.zeros((k,d))  # m1 and m2, store in 2*8 matrices
        self.S = np.zeros((k,d,d))   # S1 and S2, store in 2*(8*8) matrices
        self.p = np.zeros(2)  # p1 and p2, store in dimension 2 vectors

    # compute the parameters for both classes based on the training data
    def fit(self, Xtrain, ytrain):
        # Split the data into two parts based on the labels
        Xtrain1, Xtrain2 = splitData(Xtrain, ytrain)

        # Compute the parameters for each class
        # m1, S1 for class1
        self.m[0,:] = computeMean(Xtrain1)
        self.S[0,:,:] = computeCov(Xtrain1)
        # m2, S2 for class2
        self.m[1,:]  = computeMean(Xtrain2)
        self.S[1,:,:] = computeCov(Xtrain2)
        # priors for both class
        self.p = computePrior(ytrain)

    # Predictions: n (all entries will be either number 1 or 2 to denote the labels)
    def predict(self, Xtest):
        # placeholders to store the predictions
        predictions = np.zeros(Xtest.shape[0])

        # plug in the test data features and compute the discriminant functions for both classes (you need to choose the correct discriminant functions)
        discriminants1 = np.zeros(Xtest.shape[0])
        discriminants2 = np.zeros(Xtest.shape[0])
    
        m1 = self.m[0]
        m2 = self.m[1]
        S1 = self.S[0]
        S2 = self.S[1]
        
        S_det1 = np.linalg.det(S1)
        S_det2 = np.linalg.det(S2)
        S1_inv = np.linalg.inv(S1)
        S2_inv = np.linalg.inv(S2)

        for i in range(Xtest.shape[0]):
            x = Xtest[i]
            diff1 = np.reshape((x - m1), (Xtest.shape[1],1))
            diff2 = np.reshape((x - m2), (Xtest.shape[1],1))

            g1 = -0.5 * np.log(S_det1) - 0.5 * diff1.T @ S1_inv @ diff1 + np.log(self.p[0])
            g2 = -0.5 * np.log(S_det2) - 0.5 * diff2.T @ S2_inv @ diff2 + np.log(self.p[1])

        
            discriminants1[i] = g1
            discriminants2[i] = g2

        # if g1>g2, choose class1, otherwise choose class 2, can convert g1 and g2 into final predictions
        # e.g. g1 = [0.1, 0.2, 0.4, 0.3], g2 = [0.3, 0.3, 0.3, 0.4], => predictions = [2,2,1,2]
        predictions[discriminants1 >= discriminants2] = 1
        predictions[discriminants1 < discriminants2] = 2
        return predictions


class GaussianDiscriminant_C2:
    # classifier initialization
    # input:
    #   k: number of classes (2 for this assignment)
    #   d: number of features; feature dimensions (8 for this assignment)
    def __init__(self, k=2, d=8):
        self.m = np.zeros((k,d))  # m1 and m2, store in 2*8 matrices
        self.S = np.zeros((k,d,d))  # S1 and S2, store in 2*(8*8) matrices
        self.shared_S =np.zeros((d,d))  # the shared convariance S that will be used for both classes 8*8
        self.p = np.zeros(2)  # p1 and p2, store in dimension 2 vectors

    # compute the parameters for both classes based on the training data
    def fit(self, Xtrain, ytrain):
        # Split the data into two parts based on the labels
        Xtrain1, Xtrain2 = splitData(Xtrain, ytrain)

        # Compute the parameters for each class
        # m1, S1 for class1
        self.m[0,:] = computeMean(Xtrain1)
        self.S[0,:,:] = computeCov(Xtrain1)
        # m2, S2 for class2
        self.m[1,:]  = computeMean(Xtrain2)
        self.S[1,:,:] = computeCov(Xtrain2)
        # priors for both class
        self.p = computePrior(ytrain)

        # Compute the shared covariance matrix that is used for both class
        # shared_S = p1*S1+p2*S2
        self.shared_S = self.p[0]*self.S[0] + self.p[1]*self.S[1]
        

    # predict the labels for test data
    # Input:
    # Xtest: n*d
    # Output:
    # Predictions: n (all entries will be either number 1 or 2 to denote the labels)
    def predict(self, Xtest):
        predictions = np.zeros(Xtest.shape[0])

        discriminants1 = np.zeros(Xtest.shape[0])
        discriminants2 = np.zeros(Xtest.shape[0])

        for i in range(Xtest.shape[0]):
            x = Xtest[i]
            diff1 = np.reshape((x - self.m[0]), (Xtest.shape[1],1))
            diff2 = np.reshape((x - self.m[1]), (Xtest.shape[1],1))

            g1 = - 0.5 * diff1.T @ (np.linalg.inv(self.shared_S)) @ diff1 + np.log(self.p[0])
            g2 = - 0.5 * diff2.T @ (np.linalg.inv(self.shared_S)) @ diff2 + np.log(self.p[1])

            discriminants1[i] = g1
            discriminants2[i] = g2

        # if g1>g2, choose class1, otherwise choose class 2, convert g1 and g2 into final predictions
        # e.g. g1 = [0.1, 0.2, 0.4, 0.3], g2 = [0.3, 0.3, 0.3, 0.4], => predictions = [2,2,1,2]
        predictions[discriminants1 > discriminants2] = 1
        predictions[discriminants1 <= discriminants2] = 2

        return predictions


class GaussianDiscriminant_C3:
    # classifier initialization
    # input:
    #   k: number of classes (2 for this assignment)
    #   d: number of features; feature dimensions (8 for this assignment)
    def __init__(self, k=2, d=8):
        self.m = np.zeros((k,d))  # m1 and m2, store in 2*8 matrices
        self.S = np.zeros((k,d,d))  # S1 and S2, store in 2*(8*8) matrices
        self.shared_S =np.zeros((d,d))  # the shared convariance S that will be used for both classes
        self.p = np.zeros(2)  # p1 and p2, store in dimension 2 vectors

    # compute the parameters for both classes based on the training data
    def fit(self, Xtrain, ytrain):
        # Split the data into two parts based on the labels
        Xtrain1, Xtrain2 = splitData(Xtrain, ytrain)

        # Compute the parameters for each class
        # m1, S1 for class1
        self.m[0,:] = computeMean(Xtrain1)
        self.S[0,:,:] = computeCov(Xtrain1)
        # m2, S2 for class2
        self.m[1,:]  = computeMean(Xtrain2)
        self.S[1,:,:] = computeCov(Xtrain2)
        # priors for both class
        self.p = computePrior(ytrain)
        self.S[0] = np.diag(np.diag(self.S[0]))
        self.S[1] = np.diag(np.diag(self.S[1]))

        self.shared_S = self.p[0]*self.S[0] + self.p[1]*self.S[1]


    # predict the labels for test data
    # Input:
    # Xtest: n*d
    # Output:
    # Predictions: n (all entries will be either number 1 or 2 to denote the labels)
    def predict(self, Xtest):
        predictions = np.zeros(Xtest.shape[0])

        # if g1>g2, choose class1, otherwise choose class 2, convert g1 and g2 into final predictions
        # e.g. g1 = [0.1, 0.2, 0.4, 0.3], g2 = [0.3, 0.3, 0.3, 0.4], => predictions = [2,2,1,2]

        discriminants1 = np.zeros(Xtest.shape[0])
        discriminants2 = np.zeros(Xtest.shape[0])
        for i in range(Xtest.shape[0]):
            x = Xtest[i]
            g1 = 0
            g2 = 0
            for j in range(Xtest.shape[1]):
                g1 += (x[j] - self.m[0][j]) ** 2 / np.diag(self.shared_S)[j]
                g2 += (x[j] - self.m[1][j]) ** 2 / np.diag(self.shared_S)[j]
            g1 = g1*(-0.5) + np.log(self.p[0])
            g2 = g2*(-0.5) + np.log(self.p[1])

            discriminants1[i] = g1
            discriminants2[i] = g2

        predictions[discriminants1 > discriminants2] = 1
        predictions[discriminants1 <= discriminants2] = 2

        return predictions



# Input:
# features: n*d matrix (n is the number of samples, d is the number of dimensions of the feature)
# labels: n vector
# Output:
# features1: n1*d
# features2: n2*d
# n1+n2 = n, n1 is the number of class1, n2 is the number of samples from class 2
def splitData(features, labels):
    # features1 = np.zeros([np.sum(labels == 1),features.shape[1]])  
    # features2 = np.zeros([np.sum(labels == 2),features.shape[1]])
    # if features = [[1,1],[2,2],[3,3],[4,4]] and labels = [1,1,1,2], the resulting feature1 and feature2 will be
    # feature1 = [[1,1],[2,2],[3,3]], feature2 = [[4,4]]
    features1 = features[labels == 1]
    features2 = features[labels == 2]

    return features1, features2


# compute the mean of input features
# input: 
# features: n*d
# output: d
def computeMean(features):
    m = np.mean(features, axis=0)
    return m


# compute the covariance of input features
# input: 
# features: n*d
# output: d*d
def computeCov(features):
    # S = np.eye(features.shape[1])
    S = np.cov(features.T)

    return S


# compute the priors of input features
# input: 
# features: n
# output: 2
def computePrior(labels):
    p = np.array([0.5,0.5])
    p1 = np.sum(labels==1, axis=0) / labels.shape[0]
    p2 = np.sum(labels==2, axis=0) / labels.shape[0]
    return p1, p2
