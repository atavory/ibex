from __future__ import absolute_import

import inspect
import types
import functools
import copy

import six
import numpy as np
import pandas as pd
from sklearn import base
from sklearn import pipeline

from ._frame_mixin import FrameMixin


def frame(step):
    if isinstance(step, pipeline.Pipeline):
        return frame(pipeline.Pipeline)(steps=step.steps)

    if not inspect.isclass(step):
        f = frame(type(step))()
        params = step.get_params()
        f.set_params(params)
        return f

<<<<<<< HEAD
    class _Base(step, FrameMixin):
        def __init__(self, *args, **kwargs):
            kwargs = kwargs.copy()
            if 'columns' in kwargs:
                FrameMixin.__init__(self, columns=kwargs['columns'])
                del kwargs['columns']
            super(_Base, self).__init__(*args, **kwargs)

        def set_params(self, **params):
            if 'columns' in params:
                FrameMixin.set_params(self, columns=params['columns'])
                params = params.copy()
                del params['columns']

            super(_Base, self).set_params(**params)

        def get_params(self, deep=True):
            mixin_params = FrameMixin.get_params(self, deep=deep)
            wrapped_params = super(_Base, self).get_params(deep=deep)
            mixin_params.update(wrapped_params)
            return mixin_params
=======
    class _Adapter(step, FrameMixin):
        def __repr__(self):
            return step.__repr__(self).replace('_Adapter', 'Adapter[' + step.__name__ + ']', 1)

        def __str__(self):
            return step.__str__(self).replace('_Adapter', 'Adapter[' + step.__name__ + ']', 1)
>>>>>>> temp-branch

        def fit(self, X, *args):
            FrameMixin.set_params(self, columns=X.columns)

            res = super(_Base, self).fit(X, *args)

            return self.__process_wrapped_call_res(X, res)

        def predict(self, X, *args):
            res = super(_Base, self).predict(self.__x(X), *args)

            return self.__process_wrapped_call_res(X[FrameMixin.get_params(self)['columns']], res)

        def transform(self, X, *args):
            res = super(_Base, self).transform(self.__x(X), *args)

            return self.__process_wrapped_call_res(X[FrameMixin.get_params(self)['columns']], res)

        # Tmp Ami - should be in base?
        def __x(self, X):
            # Tmp Ami - should be in base?
            X = X[FrameMixin.get_params(self)['columns']]
            # Tmp Ami - is_subclass or isinstance?
            return X if FrameMixin.is_subclass(self) else X.as_matrix()

        def __process_wrapped_call_res(self, X, res):
            if isinstance(res, np.ndarray):
                if len(res.shape) == 1:
                    return pd.Series(res, index=X.index)
                if len(res.shape) == 2:
                    if len(X.columns) == res.shape[1]:
                        columns = X.columns
                    else:
                        columns = range(res.shape[1])
                    return pd.DataFrame(res, index=X.index, columns=columns)

            return res

    argspec = inspect.getargspec(step.__init__)
    formatted_args = inspect.formatargspec(*argspec)
    if ',' in formatted_args:
        formatted_args = formatted_args.replace(')', ', columns=None)')
    else:
        formatted_args = '(columns=None)'
    print(formatted_args)

    adapter_code = """
import numpy as np
inf = np.inf

class _Adapter(_Base):
    __name__ = step.__name__
    __doc__ = step.__doc__

    def __init__%s:
        _Base.__init__(self) # %s
    """ % (formatted_args, formatted_args)

    print(adapter_code)

    six.exec_(adapter_code, locals())
    return locals()['_Adapter']
