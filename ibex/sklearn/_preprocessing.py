from __future__ import absolute_import


from .._function_transformer import FunctionTransformer as PdFunctionTransformer


def update_module(module):
    setattr(module, 'FunctionTransformer', PdFunctionTransformer)
