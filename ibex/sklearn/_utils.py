from __future__ import absolute_import


def get_matching_estimators(module, base):
    ests = []
    for name in dir(module):
        c = getattr(module, name)
        if base is not None:
            try:
                if not issubclass(c, base):
                    continue
            except TypeError:
                continue
        ests.append(c)
    return ests
