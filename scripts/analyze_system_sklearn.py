import os
import importlib
import inspect

import six
import sklearn
from sklearn import base


m_names = set()
js_c_names = [{'id': 'sklearn', 'value': ''}]
for mod_name in sklearn.__all__:
    js_c_names += [{'id': 'sklearn.' + mod_name, 'value': ''}]
    import_str = 'from sklearn import %s as _orig' % mod_name
    exec(import_str)
    for name in dir(_orig):
        c = getattr(_orig, name)
        try:
            if not issubclass(c, base.BaseEstimator):
                continue
        except TypeError:
            continue
        js_c_names += [{'id': 'sklearn.' + mod_name + '.' + name, 'value': '1'}]
        for m_name in c.__dict__:
            if m_name.startswith('_'):
                continue
            m = getattr(c, m_name)
            if not six.callable(m):
                continue
            sig = inspect.signature(m)
            params = list(sig.parameters)
            if params[: 2] != ['self', 'X']:
                continue
            m_names.add(m_name + '-' + ','.join(params[: 3]))


print('\n\nmodules:\n\n')
print('\n'.join(sorted(sklearn.__all__)))

print('\n\nmethods:\n\n')
print('\n'.join(sorted(m_names)))

print('\n\js_c_names:\n\n')
print('[')
print(',\n'.join([('\t' + str(d)) for d in js_c_names]))
print(']')


