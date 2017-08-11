from __future__ import absolute_import


import functools
import inspect

import pandas as pd
from sklearn import base
from .._adapter import  frame


_extra_doc = """

.. note::

    Bumpty boop


"""


def update_module(name, module):

    if name != 'feature_selection':
        return

    for name in dir(module):
        c = getattr(module, name)
        try:
            if issubclass(c, base.TransformerMixin):
                module.__doc__ += _extra_doc
                break
        except TypeError:
            pass





