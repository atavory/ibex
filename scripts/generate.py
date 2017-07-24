import os
import importlib
import inspect

import six
import sklearn
from sklearn import base
import jinja2


m_names = set()
for mod_name in sklearn.__all__:
    import_str = 'from sklearn import %s as _orig' % mod_name
    exec(import_str)
    for name in dir(_orig):
        c = getattr(_orig, name)
        try:
            if not issubclass(c, base.BaseEstimator):
                continue
        except TypeError:
            continue
        for m_name in c.__dict__:
            if m_name.startswith('_'):
                continue
            if m_name == 'score':
                continue
            m = getattr(c, m_name)
            if not six.callable(m):
                continue
            sig = inspect.signature(m)
            params = list(sig.parameters)
            if params[: 2] != ['self', 'X']:
                continue
            # print(name, c, m_name, params)
            m_names.add(m_name)

loader = jinja2.FileSystemLoader(os.path.dirname(__file__))
env = jinja2.Environment(loader=loader)
tmpl = env.get_template('_adapter.py.jinja2')
cnt =tmpl.render(
	comment='# Auto generted from _adapter.py.jinja2',
	m_names=m_names)
open(os.path.join(os.path.dirname(__file__), '../ibex/_adapter.py'), 'w').write(cnt)
