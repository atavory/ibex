"""
Pandas adapters for sklearn.
"""


from ._frame_mixin import FrameMixin
from ._adapter import frame
from ._feature_union import _FeatureUnion
from ._function_transformer import trans
import sklearn


__all__ = []

__version__ = '0.1.0'

__all__ += ['__version__']

__all__ += ['FrameMixin']

__all__ += ['frame']

__all__ += ['trans']

__all__ += ['sklearn']

