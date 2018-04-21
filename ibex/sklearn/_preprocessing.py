from __future__ import absolute_import


import inspect

import pandas as pd
from sklearn import base
from sklearn import decomposition as orig

from .._adapter import frame_ex
from .._utils import (
    get_lowest_level_column_names,
    set_lowest_level_column_names
)


def polynomial_feautures_transform_imp(self, base_ret):
    def pow_str(c, p):
        if p == 0:
            return ''
        if p == 1:
            return c
        return c + '^' + str(p)

    cols = []
    for power in self.powers_:
        label = ' '.join([pow_str(c, p) for c, p in zip(self.x_columns, power) if pow_str(c, p)])
        if not label:
            label = '1'
        cols.append(label)
    set_lowest_level_column_names(base_ret, cols)

    return base_ret


polynomial_feautures_transform_imp_doc = """
foo
"""
