"""FrequencyCreator.py, author=LoganF
Utility functions to allow for the creation of commonly needed frequencies"""
import numpy as np


def logspace(start, stop, step):
    """Generates n=step log spaced frequencies starting at start and ending at stop

    Parameters
    ----------
    start: int, frequency to start at
    stop: int, frequency to stop at
    step: int, number of frequencies desired between start and stop

    Returns
    -------
    log spaced frequencies between start and stop
    """
    return np.logspace(np.log10(start), np.log10(stop), step)


def linspace(start, stop, step):
    """Generates n=step linear spaced frequencies starting at start and ending at stop

    Parameters
    ----------
    start: int, frequency to start at
    stop: int, frequency to stop at
    step: int, number of frequencies desired between start and stop

    Returns
    -------
    linear spaced frequencies between start and stop
    """
    return np.linspace(np.log10(start), np.log10(stop), step)