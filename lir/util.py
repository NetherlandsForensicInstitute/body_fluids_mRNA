import numpy as np


def Xn_to_Xy(*Xn):
    """
    Convert Xn to Xy format.

    Xn is a format where samples are divided into separate variables based on class.
    Xy is a format where all samples are concatenated, with an equal length variable y indicating class."""
    X = np.concatenate(Xn)
    y = np.concatenate([np.ones((X.shape[0],), dtype=np.int8) * i for i, X in enumerate(Xn)])
    return X, y


def Xy_to_Xn(X, y):
    """
    Convert Xy to Xn format.

    Xn is a format where samples are divided into separate variables based on class.
    Xy is a format where all samples are concatenated, with an equal length variable y indicating class."""
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)
    assert X.shape[0] == y.shape[0], 'dimensions of X and y do not match'
    assert y.shape[1] == 1
    y_uniq = np.unique(y)
    assert len(y_uniq) == 2, 'expected two classes; found: {}'.format(y_uniq)
    return [X[(y == yvalue).reshape(-1)] for yvalue in y_uniq]