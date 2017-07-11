import sys


_py3 = sys.version_info[0] == 3


def _is_str(s):
    if _py3:
        return isinstance(s, str)
    return isinstance(s, basestring)


