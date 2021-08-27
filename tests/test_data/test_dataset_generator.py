"""Produces datasets for use in testing.

These datasets are focused on producing, with various amounts of
noise, a classification or regression dataset.

These datasets represent some notional metadata (accurate or not)
concerning some samples of data.
"""
import modin.pandas as pd
import pandas._typing as pdtypes
import pytest
# from numpy.random import default_rng
from sklearn.datasets import make_classification
from typing import Optional

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

        binary_classification: Makes a dataset with numeric features
        and a binary target.
    """
    if problem_type == 'binary_classification':
        features, target = make_classification(**kwargs)
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
        'binary_classification',
        dict(
            n_samples=100, n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=0
        ),
        np.array([-0.201, 21.985])
    )
]

@pytest.mark.parametrize('problem_set, kwargs, expected_data', test_cases)
def test_generate_metadata_dataset(problem_set, kwargs, expected_values):
    dataset = generate_metadata_dataset(problem_type, **kwargs)
    actual_values = df.agg({'0':'mean','18':'sum'}).round(3).values
    
