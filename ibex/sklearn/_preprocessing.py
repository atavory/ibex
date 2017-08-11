from __future__ import absolute_import


from .._function_transformer import FunctionTransformer as PDFunctionTransformer


def update_module(name, module):
    setattr(module, 'FunctionTransformer', PDFunctionTransformer)




