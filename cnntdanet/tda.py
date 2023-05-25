import numpy as np

from gtda.pipeline import Pipeline
from gtda.homology import CubicalPersistence
from gtda.diagrams import Scaler, PersistenceLandscape, PersistenceImage, BettiCurve

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ChannelLast(BaseEstimator, TransformerMixin):
    """
    Transpose given vectorizations into the channel-last convention, which is compatible with TensorFlow

    For example, 
        - (N, C, n_bins, n_bins) ndarry produced by PersistenceImage is transformed into (N, n_bins, n_bins, C)
        - (N, C, length) ndarray produced by either PersistenceLandscape or BettiCurve is transformed into (N, length, C)
    """ 
    def fit(self, X, y=None):
        self._is_fitted_ = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)

        if X.ndim == 3:
            return np.transpose(X, (0, 2, 1))

        elif X.ndim == 4:
            return np.transpose(X, (0, 2, 3, 1))


def get_tda_pipeline(method, n_jobs=-1, **kwargs):
    """
    Create gtda Pipeline that takes a mini-batch of images and generates the topological features. 

    parameters
    ----------
    method: str, the name of a vectorization method that is one of 
        ['persistence-image', 'betti-curve', and 'persistence-landscape']
        
    **kwargs: Arguments for each vectorization method.
        - If 'method'='persistence-image', 'n_bins' must be specified.
        - If 'method'='betti-curve', 'n_bins' must be specified.
        - If 'method'='persistence-landscape', 'n_layers' and 'n_bins' must be specified

    returns
    ----------
    pipeline: gtda.pipeline.Pipeline
    """
    if method not in ['persistence-image', 'betti-curve', 'persistence-landscape']:
        raise ValueError(f"'method' should be one of ['persistence-image', 'betti-curve', 'persistence-landscape']")

    pd = CubicalPersistence(
        homology_dimensions=[0, 1],
        coeff=2,
        periodic_dimensions=None,
        infinity_values=None,
        reduced_homology=True,
        n_jobs=n_jobs
    )    
    scaler = Scaler(n_jobs=n_jobs)
    pipeline = Pipeline([("Diagram", pd), ("Scaler", scaler)])

    if method == 'persistence-image':
        if 'n_bins' not in kwargs:
            raise ValueError(f"Argument 'n_bins' must be given.")
        pipeline.steps.append(("Vectorization", PersistenceImage(n_bins=kwargs['n_bins'], n_jobs=n_jobs)))

    elif method == 'betti-curve':
        if 'n_bins' not in kwargs:
            raise ValueError(f"Argument 'n_bins' must be given.")
        pipeline.steps.append(("Vectorization", BettiCurve(n_bins=kwargs['n_bins'], n_jobs=n_jobs)))

    elif method == 'persistence-landscape':
        if ('n_bins' not in kwargs) or (('n_layers' not in kwargs)):
            raise ValueError(f"Arguments 'n_bins' and 'n_layers' must be given.")

        pipeline.steps.append(("Vectorization", PersistenceLandscape(n_layers=kwargs['n_layers'], n_bins=kwargs['n_bins'], n_jobs=n_jobs)))

    pipeline.steps.append(("Reshape", ChannelLast()))

    return pipeline
