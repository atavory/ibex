from __future__ import absolute_import


from .._function_transformer import FunctionTransformer as PDFunctionTransformer


def update_module(name, module):
    if name != 'preprocessing':
        return

    setattr(module, 'FunctionTransformer', PDFunctionTransformer)




