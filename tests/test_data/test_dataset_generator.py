"""Produces datasets for use in testing.

These datasets are focused on producing, with various amounts of
noise, a classification or regression dataset.

These datasets represent some notional metadata (accurate or not)
concerning some samples of data.
"""
import modin.pandas as pd
import numpy as np
import pandas._typing as pdtypes
import pytest
# from numpy.random import default_rng
from sklearn.datasets import make_classification, make_regression
from typing import Optional, Union

def generate_metadata_dataset(problem_type:str, **kwargs) -> pdtypes.FrameOrSeries:
    """Generates a sample metadata dataset.

    This produces a metadata dataset for a population of learners
    to explore. This does *not* produce embeddings, instead this
    simulates a customer handing over a metadata dataset. Adding
    more noise simulates having less than capable data labelers.

    Parameters
    ----------
    problem_type: str
        The style of problem you want to solve.
    kwargs
        Forwarded on to the particular dataset generator.

    Returns
    -------
    random_data: pdtypes.FrameOrSeries
        This returns a pandas dataframe with metadata dependent on
        the implementation.

        classification: Makes a dataset with numeric features
        and n classes.

        regression: Make a regression dataset

    Examples
    --------
    >>> metadata_dataset = generate_metadata_dataset('classification',random_state=0)
    >>> metadata_dataset.describe().T[['mean', 'std']].round(2).head(3)
       mean   std
    0 -0.20  0.92
    1 -0.07  1.07
    2 -0.02  0.98
    >>> metadata_dataset = generate_metadata_dataset('regression',random_state=0)
    >>> metadata_dataset.describe().T[['mean', 'std']].round(2).head(3)
       mean   std
    0  0.04  0.96
    1 -0.24  0.91
    2  0.07  0.96
    """
    if problem_type == 'classification':
        features, target = make_classification(**kwargs)
        df = pd.DataFrame(features)
        df.columns = [str(x) for x in df.columns]
        return df.assign(y=target)
    elif problem_type == 'regression':
        features, target = make_regression(**kwargs)
        df = pd.DataFrame(features)
        df.columns = [str(x) for x in df.columns]
        return df.assign(y=target)
    else:
        err_msg = f"""Dataset Generation Error:

        Unable to produce a dataset for problem type.
        If you are very interested in a new problem type, why not
        fill out an issue?

        Problem Type
        ------------\n\t{problem_type}
        """
        raise NotImplementedError(err_msg)

test_cases = [
    (
        'classification',
        dict(
            n_samples=100, n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=0
        ),
        np.array([-0.201, 21.985])
    ),
    (
        'regression',
        dict(n_samples=100, n_features=100, n_informative=10, n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.0, shuffle=True, coef=False, random_state=None),
        np.array([-.002, -6.57])
    )
]

@pytest.mark.parametrize('problem_set, kwargs, expected_data', test_cases)
def test_generate_metadata_dataset(problem_set, kwargs, expected_data):
    dataset = generate_metadata_dataset(problem_set, **kwargs)
    actual_values = dataset.agg({'0':'mean','18':'sum'}).round(3).values


def generate_random_embeddings(
    embedding_type: str,
    n: Union[pdtypes.FrameOrSeries, int],
    d: Optional[int] = 2,
    **kwargs
) -> pdtypes.FrameOrSeries:
    """Generate shaped pseudo-random embeddings.

    This creates a fresh set of embeddings for a given dataset, or
    of a certain size. These embeddings take a specific shape
    dependent on the embedding type desired.

    * Uniform: Uniform embedding will uniformly randomly disperse
      the embedded points within a `d` dimensional unit hypercube.
      This algorithm accepts `d` as a keyword argument
    * Clusters: Cluster embedding will randomly disperse `k`
      centroids within a `d` dimensional unit hypercube. All instances
      of metadata will then be gaussian distributed around those
      centroids.

    """