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
from numpy.random import default_rng
from sklearn.datasets import make_classification, make_regression, make_blobs
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
    """Test generate metadata dataset."""
    dataset = generate_metadata_dataset(problem_set, **kwargs)
    actual_values = dataset.agg({'0':'mean','18':'sum'}).round(3).values


def fake_learners(
    c: int = 3,
    n: Union[int, ArrayLike] = 100,
    d: int = 2,
    seed: Optional[int] = None,
):
    """Draw fake learners.

    This draws n learners at random.
    
    These learners come in the form of [d + 1] wide arrays stacked
    on top of each others, resulting in a [n x (d + 1)] sized array.
    
    These learners are drawn from c independent d-dimensional
    gaussian centroids, as a hand-wavy representation of *archetypes*
    of learners. Think of this as one cluster being visual learners
    while another cluster would represent textual learners.
    The first [d] components in every 'row' along axis 0 represent
    the embedding location.

    Parameters
    ----------
    c: int = 3
        The number of learner classes to draw from.
    n: [int, ArrayLike] = 100
        The number of students to draw, or the number to draw by
        class.
    d: int = 2
        The dimensionality of the student embeddings.
    seed: Optional[int] = None
        Overload the pseudo RNG.

    Returns
    -------
    student_data: ArrayLike
        An [n x (d + 1)] array of embeddings drawn from a learner space.

    Examples
    --------
    >>> import pandas as pd
    >>> silly_example = pd.DataFrame(fake_learners(
    ...         c=4,
    ...         n=100,
    ...         d=2,
    ...         seed=0
    ... ), columns=['x', 'y', 'class'], dtypes=['float','float','int])
    >>> silly_example.describe()
                    0           1           2
    count  100.000000  100.000000  100.000000
    mean     0.020995    4.096962    1.500000
    std      1.745476    2.605814    1.123666
    min     -3.018162   -0.828619    0.000000
    25%     -1.596922    2.112942    0.750000
    50%     -0.066254    3.864037    1.500000
    75%      1.253925    5.884486    2.250000
    max      3.938418    9.764992    3.000000
    """
    x, y = make_blobs(
        n_samples=n,
        n_features=d,
        centers=c,
        random_state=seed,
    )
    return np.concatenate([x, y.reshape(-1, 1)],axis=1)


def test_fake_learners():
    """Test fake_learners."""
    expected_data = np.array([[ 0.8753, 4.3457],
                              [ 1.9285, 1.0336],
                              [-1.4318, 3.3165],
                              [-1.288 , 7.6921]])
    calculated_data = pd.DataFrame(fake_learners(n=100,c=4,d=2,seed=0)).groupby(2).mean().round(4).values
    assert np.allclose(
        expected_data,
        calculated_data
    )


def fake_tests(
    n: int = 3,
    d: int = 2,
    seed: Optional[int] = None,
):
    """Draw fake tests.

    This draws n tests uniformly at random.
    
    These tests come in the form of [d] wide arrays stacked
    on top of each others, resulting in a [n x d] sized array.
    
    These tests are drawn uniformly at random in d dimensions,
    producing d-dimensional embedding vectors.

    Parameters
    ----------
    n: int = 3
        The number of tests to draw, or the number to draw by class.
    d: int = 2
        The dimensionality of the test embeddings.
    seed: Optional[int] = None
        Overload the pseudo RNG.

    Returns
    -------
    test_data: ArrayLike
        An [n x d] array of embeddings drawn from a test space.

    Examples
    --------
    >>> import numpy as np
    >>> np.round(fake_tests(
    ...         n=3,
    ...         d=2,
    ...         seed=0
    ... ),3)
    array([[0.637 , 0.2698],
           [0.041 , 0.0165],
           [0.8133, 0.9128]])
    """
    rng = default_rng(seed)
    return rng.uniform(low=0, high=1, size=(n, d))


def test_fake_tests():
    """Test fake_tests."""
    expected_data = np.array([[0.637 , 0.2698],
                              [0.041 , 0.0165],
                              [0.8133, 0.9128]])
    calculated_data = np.round(fake_tests(n=3,d=2,seed=0),4)
    assert np.allclose(
        expected_data,
        calculated_data
    )

