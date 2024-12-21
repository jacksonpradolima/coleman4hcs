"""
Effect Size Computation for Statistical Analysis

This module implements methods for computing and analyzing effect sizes, specifically
Vargha and Delaney's A index, which measures the probability that one random observation
from a treatment group is larger than a random observation from a control group. The A index
is widely used in educational and behavioral statistics.

Functions:
- VD_A: Computes the Vargha and Delaney A index for two groups.
- VD_A_DF: Computes pairwise A index comparisons for multiple groups in a pandas DataFrame.
- reduce: Filters and annotates effect sizes for comparisons against a specified best group.

Key Features:
- Accurate computation of the A index using a formula that minimizes numerical errors.
- Categorization of effect sizes into magnitudes (negligible, small, medium, large).
- Flexible handling of data through pandas DataFrame operations for group comparisons.
- Latex-compatible symbols for presenting effect sizes in reports or visualizations.

References:
- A. Vargha and H. D. Delaney, "A critique and improvement of the CL common language
  effect size statistics of McGraw and Wong," Journal of Educational and Behavioral
  Statistics, vol. 25, no. 2, pp. 101–132, 2000.
- Hess and Kromrey, 2004, for thresholds of effect size magnitudes.
"""
import itertools as it
from bisect import bisect_left
from typing import List

import numpy as np
import pandas as pd
import scipy.stats as ss
from pandas import Categorical

# turn off the SettingWithCopyWarning
pd.set_option('mode.chained_assignment', None)


def VD_A(treatment: List[float], control: List[float]):
    """
    Computes Vargha and Delaney A index
    A. Vargha and H. D. Delaney.
    A critique and improvement of the CL common language
    effect size statistics of McGraw and Wong.
    Journal of Educational and Behavioral Statistics, 25(2):101-132, 2000

    The formula to compute A has been transformed to minimize accuracy errors
    See: http://mtorchiano.wordpress.com/2014/05/19/effect-size-of-r-precision/

    :param treatment: a numeric list
    :param control: another numeric list

    :returns the value estimate and the magnitude
    """
    m = len(treatment)
    n = len(control)

    if m != n:
        raise ValueError("Data d and f must have the same length")

    r = ss.rankdata(treatment + control)
    r1 = sum(r[0:m])

    # Compute the measure
    # A = (r1/m - (m+1)/2)/n # formula (14) in Vargha and Delaney, 2000
    A = (2 * r1 - m * (m + 1)) / (2 * n * m)  # equivalent formula to avoid accuracy errors

    levels = [0.147, 0.33, 0.474]  # effect sizes from Hess and Kromrey, 2004
    magnitude = ["negligible", "small", "medium", "large"]
    scaled_A = (A - 0.5) * 2

    magnitude = magnitude[bisect_left(levels, abs(scaled_A))]
    estimate = A

    return estimate, magnitude


def VD_A_DF(data, val_col: str = None, group_col: str = None, sort=True):
    """

    :param data: pandas DataFrame object
        An array, any object exposing the array interface or a pandas DataFrame.
        Array must be two-dimensional. Second dimension may vary,
        i.e. groups may have different lengths.
    :param val_col: str, optional
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains values.
    :param group_col: str, optional
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains group names.
    :param sort : bool, optional
        Specifies whether to sort DataFrame by group_col or not. Recommended
        unless you sort your data manually.

    :return: stats : pandas DataFrame of effect sizes

    Stats summary ::
    'A' : Name of first measurement
    'B' : Name of second measurement
    'estimate' : effect sizes
    'magnitude' : magnitude

    """

    x = data.copy()
    if sort:
        x[group_col] = Categorical(x[group_col], categories=x[group_col].unique(), ordered=True)
        x.sort_values(by=[group_col, val_col], ascending=True, inplace=True)

    groups = x[group_col].unique()

    # Pairwise combinations
    g1, g2 = np.array(list(it.combinations(np.arange(groups.size), 2))).T

    # Compute effect size for each combination
    ef = np.array([VD_A(list(x[val_col][x[group_col] == groups[i]].values),
                        list(x[val_col][x[group_col] == groups[j]].values)) for i, j in zip(g1, g2)])

    return pd.DataFrame({
        'base': np.unique(data[group_col])[g1],
        'compared_with': np.unique(data[group_col])[g2],
        'estimate': ef[:, 0],
        'magnitude': ef[:, 1]
    })


def reduce(df, best, symbols=True):
    """
    Reduce a pandas DataFrame of effect sizes to compare only against to the best among the comparison (algorithm/item)

    :param df: pandas DataFrame object
        A pandas DataFrame of effect sizes
    :param best: str
        The name of the best sample, for instance, algorithm AAA
    :param symbols: bool, optional
        Specifies whether to use symbols from LaTeX to represent the magnitudes
    :return:
    """
    # Effect Size
    magnitude = ["negligible", "small", "medium", "large"]
    symbols = ["$\\blacktriangledown$", "$\\triangledown$", "$\\vartriangle$", "$\\blacktriangle$"]

    # Get only the effect size manitudes related with the best
    df = df[(df.base == best) | (df.compared_with == best)]

    # Create a new column to compare against the other policies
    df['temp'] = df.apply(lambda row: row['compared_with'] if row['base'] == best else row['base'], axis=1)

    # Get magnitude symbol (in latex) for each comparison
    # The best has the bigstart symbol
    df['effect_size_symbol'] = df['temp'].apply(lambda x: "$\\bigstar$"
    if x == best
    else symbols[magnitude.index(df.loc[df['temp'] == x, 'magnitude'].tolist()[0])])

    df.drop(['temp'], axis=1, inplace=True)

    return df
