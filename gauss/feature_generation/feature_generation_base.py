import abc

from gauss.component import Component
from externals.multiple import MultipleMeta

class Transformer(metaclass=MultipleMeta):
    """class for all transformers in scikit-learn."""
    def __init__(self):
        pass

    def run(self, X, y=None, **fit_params):
        """Fit to data, then transform it.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)

    @abc.abstractmethod
    def fit(self, X, y=None, **kwargs):
        print(X, y)

    @abc.abstractmethod
    def fit(self, X, **kwargs):
        print(X)


test = Transformer()
test.fit(1)
test.fit(1,2)
