from sklearn import decomposition as _orig
from sklearn import base

from .. import frame


for name in dir(_orig):
	prd = getattr(_orig, name)
	try:
		if issubclass(prd, base.BaseEstimator):
			globals()[name] = frame(prd)
	except TypeError:
		continue
