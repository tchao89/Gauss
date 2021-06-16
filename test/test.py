import numpy as np
from scipy import special
from scipy.stats import chi2_contingency
from scipy.stats import chi2 as scipy_chi2
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils import check_array
from sklearn.feature_selection import SelectKBest, chi2


def chi2_(X, y):
    """Compute chi-squared stats between each non-negative feature and class.

    This score can be used to select the n_features features with the
    highest values for the test chi-squared statistic from X, which must
    contain only non-negative features such as booleans or frequencies
    (e.g., term counts in document classification), relative to the classes.

    Recall that the chi-square test measures dependence between stochastic
    variables, so using this function "weeds out" the features that are the
    most likely to be independent of class and therefore irrelevant for
    classification.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Sample vectors.

    y : array-like of shape (n_samples,)
        Target vector (class labels).

    Returns
    -------
    chi2 : ndarray of shape (n_features,)
        Chi2 statistics for each feature.

    p_values : ndarray of shape (n_features,)
        P-values for each feature.

    Notes
    -----
    Complexity of this algorithm is O(n_classes * n_features).

    See Also
    --------
    f_classif : ANOVA F-value between label/feature for classification tasks.
    f_regression : F-value between label/feature for regression tasks.
    """

    # XXX: we might want to do some of the following in logspace instead for
    # numerical stability.
    X = check_array(X, accept_sparse='csr')

    Y = LabelBinarizer().fit_transform(y)
    if Y.shape[1] == 1:
        Y = np.append(1 - Y, Y, axis=1)

    observed = safe_sparse_dot(Y.T, X)          # n_classes * n_features

    feature_count = X.sum(axis=0).reshape(1, -1)
    class_prob = Y.mean(axis=0).reshape(1, -1)
    expected = np.dot(class_prob.T, feature_count)

    return observed, chi_square(observed, expected)


def chi_square(f_obs, f_exp):
    """Fast replacement for scipy.stats.chisquare.

    Version from https://github.com/scipy/scipy/pull/2525 with additional
    optimizations.
    """
    f_obs = np.asarray(f_obs, dtype=np.float64)

    k = len(f_obs)
    # Reuse f_obs for chi-squared statistics
    chisq = f_obs
    chisq -= f_exp
    chisq **= 2
    with np.errstate(invalid="ignore"):
        chisq /= f_exp
    chisq = chisq.sum(axis=0)
    return chisq, special.chdtrc(k - 1, chisq)


iris = load_iris()
print(chi2_(X=iris.data[:, 1].reshape(-1, 1), y=iris.target.reshape(-1, 1)))

print(iris.data[:5, :])
res = chi2(X=iris.data, y=iris.target)
print(SelectKBest(chi2, k=2).fit_transform(iris.data.reshape(-1, 4), iris.target.reshape(-1, 1)).shape)

matrix = np.concatenate((iris.data[:, 0].reshape(-1, 1), iris.target.reshape(-1, 1)), axis=1)
_, _, degree_value, _ = chi2_contingency(matrix)
#  df， 矩阵的度 v = n_values * n_labels, threshold is 95%
print(scipy_chi2.isf(0.05, df=3))
