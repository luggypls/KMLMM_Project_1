"""Module kfda"""
import warnings
from scipy.sparse.linalg import eigsh
from scipy.sparse import eye
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import pairwise_kernels
from sklearn.neighbors import NearestCentroid
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.metrics.pairwise import linear_kernel, laplacian_kernel, rbf_kernel,\
                                     chi2_kernel, sigmoid_kernel, polynomial_kernel


class Kfda(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Kernel Fisher Discriminant Analysis classifier.
    Each class is represented by a centroid using projections in a
    Hilbert space.
    See https://arxiv.org/abs/1906.09436 for mathematical details.
    Parameters
    ----------
    n_components : int the amount of Fisher directions to use.
        This is limited by the amount of classes minus one.
        See the paper for further discussion of this limit.
    kernel : str, ['linear', 'poly', 'sigmoid', 'rbf','laplacian', 'chi2']
        The kernel to use.
        Use **kwds to pass arguments to these functions.
        See
        https://scikit-learn.org/stable/modules/metrics.html#polynomial-kernel
        for more details.
    robustness_offset : float
        The small value to add along the diagonal of N to gurantee
        valid fisher directions.
        Set this to 0 to disable the feature. Default: 1e-8.
    **kwds : parameters to pass to the kernel function.
    Attributes
    ----------
    centroids_ : array_like of shape (n_classes, n_samples) that
        represent the class centroids.
    classes_ : array of shape (n_classes,)
        The unique class labels
    weights_ : array of shape (n_components, n_samples) that
        represent the fisher components.
    clf_ : The internal NearestCentroid classifier used in prediction.
    """

    def __init__(self, n_components=1, 
                kernel='linear', 
                robustness_offset=1e-8,
                gamma = None,
                coef0 = 1,
                degree = 3,
                **kwds):
        self.kernel = kernel
        self.n_components = n_components
        self.kwds = kwds
        self.robustness_offset = robustness_offset
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

        if kernel is None:
            self.kernel = 'linear'

    def fit(self, X, y):
        """
        Fit the NearestCentroid model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array, shape = [n_samples]
            Target values (integers)
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        if self.n_components > self.classes_.size - 1:
            warnings.warn(
                "n_components > classes_.size - 1."
                "Only the first classes_.size - 1 components will be valid."
            )
        self.X_ = X
        self.y_ = y

        y_onehot = OneHotEncoder().fit_transform(
            self.y_[:, np.newaxis])

        
        if self.kernel in ['polynomial', 'poly']:
            K = polynomial_kernel(X, X, self.degree, self.gamma, self.coef0)
        elif self.kernel == 'rbf': 
            K = rbf_kernel(X, X, self.gamma)
        elif self.kernel == 'laplacian':
            K = laplacian_kernel(X, X, self.gamma)
        elif self.kernel == 'sigmoid':
            K = sigmoid_kernel(X, X, self.gamma, self.coef0)
        elif self.kernel == 'chi2':
            K = chi2_kernel(X, X, self.gamma)
        else:
            K = linear_kernel(X, X)


        m_classes = y_onehot.T @ K / y_onehot.T.sum(1)
        indices = (y_onehot @ np.arange(self.classes_.size)).astype('i')
        N = K @ (K - m_classes[indices])

        # Add value to diagonal for rank robustness
        N += eye(self.y_.size) * self.robustness_offset

        m_classes_centered = m_classes - K.mean(1)
        M = m_classes_centered.T @ m_classes_centered

        # Find weights
        w, self.weights_ = eigsh(M, self.n_components, N, which='LM')

        # Compute centers
        centroids_ = m_classes @ self.weights_

        # Train nearest centroid classifier
        self.clf_ = NearestCentroid().fit(centroids_, self.classes_)

        return self

    def transform(self, X):
        """Project the points in X onto the fisher directions.
        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features) to be
            projected onto the fisher directions.
        """
        check_is_fitted(self)
        if self.kernel == 'polynomial':
            return polynomial_kernel(X, self.X_, self.degree, self.gamma, self.coef0) @ self.weights_
        elif self.kernel == 'rbf': 
            return rbf_kernel(X, self.X_, self.gamma) @ self.weights_
        elif self.kernel == 'laplacian':
            return laplacian_kernel(X, self.X_, self.gamma) @ self.weights_
        elif self.kernel == 'sigmoid':
            return sigmoid_kernel(X, self.X_, self.gamma, self.coef0) @ self.weights_
        elif self.kernel == 'chi2':
            return chi2_kernel(X, self.X_, self.gamma) @ self.weights_
        else:
            return linear_kernel(X, self.X_) @ self.weights_
        

    def predict(self, X):
        """Perform classification on an array of test vectors X.
        The predicted class C for each sample in X is returned.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        C : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)

        X = check_array(X)

        projected_points = self.transform(X)
        predictions = self.clf_.predict(projected_points)

        return predictions

    def fit_additional(self, X, y):
        """Fit new classes without recomputing weights.
        Parameters
        ----------
        X : array-like of shape (n_new_samples, n_nfeatures)
        y : array, shape = [n_samples]
            Target values (integers)
        """
        check_is_fitted(self)
        X, y = check_X_y(X, y)

        new_classes = np.unique(y)

        projections = self.transform(X)
        y_onehot = OneHotEncoder().fit_transform(
            y[:, np.newaxis])
        new_centroids = y_onehot.T @ projections / y_onehot.T.sum(1)

        concatenated_classes = np.concatenate([self.classes_, new_classes])
        concatenated_centroids = np.concatenate(
            [self.clf_.centroids_, new_centroids])

        self.clf_.fit(concatenated_centroids, concatenated_classes)

        return self