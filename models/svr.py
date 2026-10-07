"""Baseline 1: support vector regression, one SVR per (timestep, PCA component).

The paper trains one multi-output SVR per timestep (1025 outputs). That is extremely slow,
so we compress each spectrogram frame to K PCA components and fit one SVR per
(timestep, component). Predictions are mapped back to 1025 bins with the inverse PCA.
"""
import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def _fit_one(kernel, degree, Cc, eps, Xs, z):
    return SVR(kernel=kernel, degree=degree, C=Cc, epsilon=eps).fit(Xs, z)


class SVRModel:
    def __init__(self, kernel="poly", n_components=32, degree=3, C=1.0, epsilon=0.01):
        self.kernel, self.k, self.degree, self.C, self.eps = kernel, n_components, degree, C, epsilon

    def fit(self, X, Y):
        N, T, F = Y.shape
        self.T, self.F = T, F
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        flat = Y.reshape(N * T, F)
        self.pca = PCA(n_components=self.k).fit(flat)
        Z = self.pca.transform(flat).reshape(N, T, self.k)
        self.models = Parallel(n_jobs=-1)(
            delayed(_fit_one)(self.kernel, self.degree, self.C, self.eps, Xs, Z[:, t, j])
            for t in range(T) for j in range(self.k))
        return self

    def predict(self, X):
        Xs = self.scaler.transform(X)
        Z = np.zeros((len(X), self.T, self.k))
        i = 0
        for t in range(self.T):
            for j in range(self.k):
                Z[:, t, j] = self.models[i].predict(Xs)
                i += 1
        Y = self.pca.inverse_transform(Z.reshape(-1, self.k)).reshape(len(X), self.T, self.F)
        return np.clip(Y, 0, 1).astype(np.float32)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
