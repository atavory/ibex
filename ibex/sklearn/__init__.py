import os
import sys


class ModuleImportUtility(object):

    @staticmethod
    def in_namespace(namespace, fullname):
        """
        Whether the given :param:`fullname` is or within the :attr:`namespace`.
        """
        if not fullname.startswith(namespace):
            return False
        nslen = len(namespace)
        return len(fullname) == nslen or fullname[nslen] == '.'

    @staticmethod
    def parent_name(fullname):
        """Get the parent name of :param:`fullname`."""
        return '.'.join(fullname.rsplit('.', 1)[:-1])

    @staticmethod
    def find_modules(namespace, name_parts, root_path):
        """
        Find the modules along :param:`name_parts` according to
        :param:`root_path`.

        :return :class:`list` of (fullname, file, filename, options) as
            :method:`imp.find_module`, or :value:`None` if not found.
        """
        try:
            ret = []
            ns = namespace
            path = root_path
            for n in name_parts:
                ns = '%s.%s' % (ns, n)
                fp, filename, options = imp.find_module(n, [path])
                ret.append((ns, fp, filename, options))
                path = filename
            return ret
        except ImportError:
            return None


class NamespaceSplitter(object):
    """Strip the parent namespace and split the subname to pieces."""

    def __init__(self, namespace):
        self.namespace = namespace
        self.cutoff = len(namespace.split("."))

    def cut(self, fullname):
        return fullname.split('.')[self.cutoff:]




class DefaultNewModuleLoader(object):
    """
    Load the requested module via standard import, or create a new module if
    not exist.
    """

    def load_module(self, fullname):
        print(fullname)
        import sys
        import imp

        class FakePackage(object):
            def __init__(self, path):
                self.__path__ = path

        # If the module has already been loaded, then we just fetch this module
        # from the import cache
        if fullname in sys.modules:
            return sys.modules[fullname]

        # Otherwise we try perform a standard import first, and if not found,
        # we create a new package as the required module
        m = None
        try:
            m = FakePackage(None)
            parts = fullname.split('.')
            for i, p in enumerate(parts, 1):
                ns = '.'.join(parts[:i])
                if ns in sys.modules:
                    m = sys.modules[ns]
                else:
                    if not hasattr(m, '__path__'):
                        raise ImportError()
                    fp, filename, options = imp.find_module(p, m.__path__)
                    m = imp.load_module(p, fp, filename, options)
                    sys.modules[ns] = m
        except ImportError:
            m = imp.new_module(fullname)
            m.__name__ = fullname
            m.__path__ = [fullname]
            m.__loader__ = self
            m.__file__ = '<dummy package "%s">' % fullname
            m.__package__ = ModuleImportUtility.parent_name(fullname)
        # Now insert the loaded module into the cache, and return the result
        sys.modules[fullname] = m
        return m


class DirModuleLoader(object):
    """
    Load the requested module under a directory (simulate the system import),
    all the intermediate modules will also be loaded.
    """

    def __init__(self, namespace, root_path):
        self.namespace = namespace
        self.root_path = root_path
        self.ns_splitter = NamespaceSplitter(namespace)

    def load_module(self, fullname):
        print(fullname)
        import imp
        name_parts = self.ns_splitter.cut(fullname)
        for (ns, fp, filename, options) in \
                ModuleImportUtility.find_modules(self.namespace, name_parts,
                                                 self.root_path):
            if ns not in sys.modules:
                sys.modules[ns] = imp.load_module(ns, fp, filename, options)
        return sys.modules[fullname]


class _ModuleFinder(object):
    def install(self):
        sys.meta_path[:] = [x for x in sys.meta_path if self != x] + [self]

    def find_module(self, fullname, path=None):
        print(fullname)
        # We should deal with all the parent packages of namespace, because
        # some of the intermediate packages may not exist, and need to be
        # created manually
        return
        if ModuleImportUtility.in_namespace(fullname, self.namespace):
            return DefaultNewModuleLoader()


loader = _ModuleFinder()
loader.install()


import numpy as np
from ibex.sklearn import ensemble
