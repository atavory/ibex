"""
Pandas adapters for sklearn-type estimators
"""


from ._base import *
from ._adapter import *
from ._function_transformer import *
import sklearn


__all__ = []

__version__ = '0.1.0'

__all__ += ['__version__']

__all__ += ['FrameMixin']

__all__ += ['frame', 'frame_ex']

__all__ += ['trans']

__all__ += ['sklearn']


def trans(func=None, in_cols=None, out_cols=None, pass_y=False, kw_args=None):
    """
    Arguments:

        func: One of:

            * ``None``
            * a callable
            * a step

        in_cols: One of:

            * ``None``
            * a string
            * a list of strings

        out_cols:

        pass_y: Boolean indicating whether to pass the ``y`` argument to

        kw_args:

    Returns:

        An :py:class:`sklearn.preprocessing.FunctionTransformer` object.

    Example:
    """

    from ibex.sklearn import preprocessing

    return preprocessing.FunctionTransformer(func, in_cols, out_cols, pass_y, kw_args)

__all__ += ['trans']

