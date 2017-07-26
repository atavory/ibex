"""
Pandas adapters for sklearn-type estimators
"""


from ._frame_mixin import FrameMixin
from ._adapter import frame
from ._feature_union import _FeatureUnion
from ._function_transformer import _FunctionTransformer
import sklearn


__all__ = []

__version__ = '0.1.0'

__all__ += ['__version__']

__all__ += ['FrameMixin']

__all__ += ['frame']

__all__ += ['trans']

__all__ += ['sklearn']


def trans(func=None, in_cols=None, out_cols=None, pass_y=False, kw_args=None):
    """
    Arguments:

        func: One of a:

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
    """

    from ibex.sklearn import preprocessing

    return preprocessing.FunctionTransformer(func, in_cols, out_cols, pass_y, kw_args)

__all__ += ['trans']

