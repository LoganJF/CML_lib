"""PandasFunctions.py, author=Loganf
A script for helpful pandas manipulation functions for behavioral events"""
import pandas as pd


def to_pandas_df(events):
    """Returns an inputted recarray as a pandas dataframe

    Parameters
    ----------
    events: np.recarray like, behavioral events generated using ptsa

    Returns
    -------
    df: pd.DataFrame, behavioral events
    """
    df = pd.DataFrame.from_records([event for event in events],
                                   columns=events.dtype.names)
    return df