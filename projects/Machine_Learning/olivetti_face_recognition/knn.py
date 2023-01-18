import numpy as np
from collections import Counter


class KNN:

    def __init__(self,k):
        self.k = k


    def fit(self, X, labels):
        self.X_tr = X
        self.L_tr = labels


    def predict(self, X):
        return np.array([self.prd(x) for x in X])
        
    def prd(self, x):

        dist = [np.linalg.norm(x - x_tr) for x_tr in self.X_tr]
        k_i = np.argsort(dist)[:self.k]
        k_labels = [self.L_tr[i] for i in k_i]
        freq = Counter(k_labels).most_common(1)[0][0]
        return freq
        

