import numpy as np


# You are going to implement functions for this file.

class Tree_node:
    """
    Data structure for nodes in the decision-tree
    """
    def __init__(self,):
        self.feature = None  # index of the selected feature (for non-leaf node)
        self.label = None  # class label (for leaf node), if not leaf node, label will be None
        self.left_child = None  # left child node
        self.right_child = None  # right child node


class Decision_tree:
    """
    Decision tree with binary features
    """
    def __init__(self,min_entropy, metric='entropy'):
        if metric == 'entropy':
            self.metric = self.entropy
        else:
            self.metric = self.gini_index
        self.min_entropy = min_entropy
        self.root = None

    def fit(self,train_x,train_y):
        # construct the decision-tree with recursion
        self.root = self.generate_tree(train_x,train_y)

    def predict(self,test_x):
        # iterate through all samples
        prediction = []
        for i in range(len(test_x)):
            cur_data = test_x[i]
            # traverse the decision-tree based on the features of the current sample
            cur_node = self.root
            while True:
                if cur_node.label != None:
                    break
                elif cur_node.feature == None:
                    print("You haven't selected the feature yet")
                    exit()
                else:
                    if cur_data[cur_node.feature] == 0:
                        cur_node = cur_node.left_child
                    else:
                        cur_node = cur_node.right_child
            prediction.append(cur_node.label)

        prediction = np.array(prediction)

        return prediction


    def generate_tree(self,data,label):
        # initialize the current tree node
        cur_node = Tree_node()
        node_entropy = self.metric(label)
        if node_entropy < self.min_entropy:
            cur_node.label = np.argmax(np.bincount(label))
            return cur_node

        selected_feature = self.select_feature(data,label)
        cur_node.feature = selected_feature
        select_x = data[:, selected_feature]
        left_x = data[select_x == 0]
        left_y = label[select_x == 0]
        right_x = data[select_x == 1]
        right_y = label[select_x == 1]
        cur_node.left_child = self.generate_tree(left_x, left_y)
        cur_node.right_child = self.generate_tree(right_x, right_y)
        return cur_node

    # return: best_feat, which is the index of the feature
    def select_feature(self,data,label):
        best_feat = 0
        min_entropy = float('inf')

        # iterate through all features and compute their corresponding entropy 
        for i in range(len(data[0])):
            # split data based on i-th feature
            split_x = data[:,i]
            left_y = label[split_x==0,]
            right_y = label[split_x==1,]

            # compute the combined entropy which weightedly combine the entropy/gini of left_y and right_y
            entropy_i = self.combined_entropy(left_y, right_y)

            # select the feature with minimum entropy (set best_feat)
            if entropy_i < min_entropy:
                min_entropy = entropy_i
                best_feat = i

        return best_feat

    # weightedly combine the entropy/gini of left_y and right_y
    # the weights are [len(left_y)/(len(left_y)+len(right_y)), len(right_y)/(len(left_y)+len(right_y))] 
    # return: result
    def combined_entropy(self,left_y,right_y):
        # compute the entropy of a potential split
        result = 0
        
        # !!! use self.metric(label) to compute the entropy/gini_index
        len_l = left_y.size
        len_r = right_y.size
        len_sum = len_l + len_r
        entropies = np.array([self.metric(left_y), self.metric(right_y)])
        weights = np.array([len_l / len_sum, len_r / len_sum])
        result = np.multiply(entropies, weights).sum()



        return result

    # compute entropy based on the labels
    # entropy = sum_i p_i*log2(p_i+1e-15) (add 1e-15 inside the log when computing the entropy to prevent numerical issue)
    def entropy(self,label):
        # compute the entropy of a tree node (add 1e-15 inside the log when computing the entropy to prevent numerical issue)
        node_entropy = 0
        if label.size == 0:
            return 0
        label = np.array(label)
        p_i = [label[label==i].size / label.size for i in range(10)]
        p_i = np.array(p_i)
        node_entropy = (-p_i * np.log2(p_i+1e-15)).sum()
        return node_entropy

    # compute gini_index based on the labels
    # gini_index = 1 - sum_i p_i^2
    def gini_index(self, label):
        gini_index = 0
        if len(label) == 0:
          return gini_index 

        p_i = [label[label==i].size / label.size for i in range(10)]
        p_sum = 0
        for prob in p_i:
          p_sum += prob*prob
        gini_index = 1 - p_sum
        return gini_index
