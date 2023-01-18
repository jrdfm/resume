import numpy as np
from scipy import linalg


class PCA:
    def __init__(self, n, whiten):
        self.n = n
        self.n_components = n
        self.mean = None
        self.whiten = whiten

    def fit(self, X):

        n_samples, n_features = X.shape
        n_components = self.n_components

        self.mean = np.mean(X, axis=0)
        X -= self.mean

        U, S, Vt = linalg.svd(X, full_matrices=False)

        # flip eigenvectors' sign
        U, Vt = self.svd_flip(U, Vt)

        components_ = Vt

        explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = S.copy()  # Store the singular values.

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = \
            explained_variance_ratio_[:n_components]
        self.singular_values_ = singular_values_[:n_components]
        self.explained_variance = [i/np.sum(self.singular_values_) for i in self.singular_values_]



    def svd_flip(self,u, v):

        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
        return u, v

    def transform(self, X):
        #X = X - self.mean
        if self.mean is not None:
            X = X - self.mean
        X_transformed = np.dot(X, self.components_.T)
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)
        return X_transformed
