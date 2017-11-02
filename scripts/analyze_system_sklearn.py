import os
import importlib
import inspect
import ast
import re

import six
import sklearn
from sklearn import base


_identifier_re = re.compile(r'[^\d\W][\w\d_]*_$')
print(_identifier_re.match('foo'))
print(_identifier_re.match('_foo'))
print(_identifier_re.match('foo_'))
print(_identifier_re.match('foo_bar'))
print(_identifier_re.match('foo_bar_'))


class ClassLister(ast.NodeVisitor):

    def __init__(self):
        self._cur_class = None
        self._cur_attr = None

        self._found = set()

    def visit_ClassDef(self, node):
        self._cur_class = node.name
        self.generic_visit(node)
        self._cur_class = None

    def visit_Attribute(self, node):
        if self._cur_class is None:
            return
        if _identifier_re.match(node.attr) is None:
            return
        self._cur_attr = node.attr
        self.generic_visit(node)
        self._cur_attr = None

    def visit_Name(self, node):
        if self._cur_attr is None:
            return
        if node.id != 'self':
            return
        self._found.add((self._cur_class, self._cur_attr))
        self.generic_visit(node)

    @property
    def found(self):
        return self._found


tree = ast.parse(open('/usr/local/lib/python3.6/site-packages/sklearn/linear_model/base.py').read())
v = ClassLister()
v.visit(tree)
print(v.found)
ff:w



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


