import numpy as np

# normalize raw data by (x-mean)/std
def normalize(x, mean=None, std=None):
    if mean is None:
        mean = np.mean(x, axis=0).reshape(1,-1)
        std = np.std(x, axis=0).reshape(1,-1)
    x = (x-mean)/(std+1e-5)
    return x, mean, std


def process_label(label):
    # label = np.array(label)
    one_hot = np.zeros([len(label),10])
    one_hot[np.arange(label.size),label] = 1
    return one_hot



def tanh(x):
    # implement the hyperbolic tangent activation function for hidden layer
    out = np.zeros_like(x)
    # preprocess x to boost the performance
    x = np.clip(x,a_min=-100,a_max=100)

    e = np.exp(x)
    out = (e-e**(-1))/(e**(-1)+e)

    return out 


# output: results of pluging the value into (e^xi)/(sum_i e^xi)
def softmax(x):
    # implement the softmax activation function for output layer
    e = np.exp(x) # n*d
    s = np.sum(e, axis=1, keepdims=True) # n,
    # out = e / np.expand_dims(s, axis=1) # n, -> n,1
    out = e / s # n, -> n,1
    return out


class MLP:
    def __init__(self,num_hid):
        # initialize the weights
        self.num_hid = num_hid
        self.lr = 5e-3 # 5e-3
        self.w = np.random.random([64,num_hid])
        self.w0 = np.random.random([1,num_hid])
        self.v= np.random.random([num_hid,10])
        self.v0 = np.random.random([1,10])

    def fit(self,train_x,train_y, valid_x, valid_y):
        # counter for recording the number of epochs without improvement
        count = 0
        best_valid_acc = 0


        # Stop the training if there is no improvment over the best validation accuracy for more than 100 iterations
        while count<=50:
            # training with all samples (full-batch gradient descents)
            # forward pass for all samples
            z, y = self.forward(train_x)
            # backward pass (backpropagation)
            # compute the gradients w.r.t. different parameters
            gra_v = self.dEdv(z, y, train_y)
            gra_v0 = self.dEdv0(y, train_y)
            gra_w = self.dEdw(z, y, train_x, train_y)
            gra_w0 = self.dEdw0(z, y, train_y)

            # update the parameters
            self.update(gra_w, gra_w0, gra_v, gra_v0)

            # evaluate on validation data
            predictions = self.predict(valid_x)
            valid_acc = np.count_nonzero(predictions.reshape(-1)==valid_y.reshape(-1))/len(valid_x)

            # compare the current validation accuracy with the best one
            if valid_acc>best_valid_acc:
                best_valid_acc = valid_acc
                count = 0
            else:
                count += 1

        return best_valid_acc

   
    # output: z intermediate output (n, num_hid) 
    #         y final output        (n,10)
    # z = tanh(xw+w0) x:(n,64), w: (64, num_hid), w0: (1, num_hid), z: (n, num_hid) 
    # y = softmax(zv+v0) v:(num_hid, 10), v0: (1, 10), y:(n,10)
    def forward(self, x):
        # placeholders
        s =  x @ self.w + self.w0 # n * d
        z = tanh(s) # n * d
        t = z @ self.v + self.v0
        y = softmax(t) # n 10
        # print(y)
        return z, y

    

    # Input: z, output of the intermediate layer (n, num_hid) 
    #        y, output of the last layer (n, 10)
    #        r, gt one-hot labels (n, 10)
    # Output: gra_v, (num_hid, 10)
    def dEdv(self, z, y, r):
        # placeholder
        out = np.zeros((self.num_hid,10))
        zT = z.T
        diff = y-r
        out = zT @ diff
        return out

    # c = np.ones(n,1)
    # gra_v0 = c.T@(y-r) or (y-r).sum(axis=0)
    def dEdv0(self, y, r):
        diff = np.subtract(y,r)
        out = diff.sum(axis=0)
        # print(out)
        return out

   
    def dEdw(self, z, y, x, r):
        # placeholder
        diff = y - r
        p = diff @ self.v.T
        out = x.T @ (p * (1 - z**2))
        return out


    def dEdw0(self, z, y, r):
        # placeholder
        out = np.zeros_like(self.w0)
        diff = y-r
        p = diff @ self.v.T
        m = p * (1 - z**2)
        out = m.sum(axis = 0)
        return out


    # e.g self.w = self.w - self.lr*gra_w
    def update(self, gra_w, gra_w0, gra_v, gra_v0):
        self.w += -self.lr * gra_w
        self.w0 += -self.lr * gra_w0
        self.v += -self.lr * gra_v
        self.v0 += -self.lr * gra_v0

        return 

    def predict(self,x):
        # generate the predicted probability of different classes
        z = tanh(x.dot(self.w) + self.w0)
        y = softmax(z.dot(self.v) + self.v0)
        # convert class probability to predicted labels
        y = np.argmax(y,axis=1)

        return y

    def get_hidden(self,x):
        # extract the intermediate features computed at the hidden layers
        z = tanh(x.dot(self.w) + self.w0)
        return z
