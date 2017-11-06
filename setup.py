import os
import sys
from setuptools import setup, Command
import subprocess


_python = 'python' + '.'.join((str(n) for n in sys.version_info[: 2]))


class _TestCommand(Command):
    user_options = [
        ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        run_str = "%s -m unittest discover test *test.py" % _python
        level = os.getenv('IBEX_TEST_LEVEL')
        os.system(run_str)


class _DocumentCommand(Command):
    user_options = [
        ('reduced-checks', None, "Don't perforam all checks (spelling, links, lint)")
        ]

    def initialize_options(self):
        self.reduced_checks = False

    def finalize_options(self):
        pass

    def run(self):
        from distutils.dir_util import copy_tree

        import sklearn
        from sklearn import base
        from jinja2 import Template

        class_template = Template(
            open(os.path.join('docs/source/api_class.rst.jinja2')).read())

        sklearn_modules =  {}
        for mod_name in sklearn.__all__:
            try:
                orig = __import__('sklearn.%s' % mod_name, fromlist=[''])
            except ModuleNotFoundError:
                continue

            sklearn_modules[mod_name] = []

            for name in dir(orig):
                c = getattr(orig, name)
                try:
                    if not issubclass(c, base.BaseEstimator):
                        continue
                except TypeError:
                    continue
                sklearn_modules[mod_name].append('ibex.sklearn.%s.%s' % (mod_name, name))
                content = class_template.render(
                    class_name=name,
                    full_class_name='ibex.sklearn.%s.%s' % (mod_name, name))
                f_name = 'docs/source/api_ibex_sklearn_%s_%s.rst' % (mod_name, name.lower())
                open(f_name, 'w').write(content)

        class_template = Template(
            open(os.path.join('docs/source/api.rst.jinja2')).read())
        content = class_template.render(
            sklearn_modules=sklearn_modules)
        f_name = 'docs/source/api.rst'
        open(f_name, 'w').write(content)

        run_str = "make html"
        if not self.reduced_checks:
            run_str += ' spelling lint linkcheck'
        subprocess.check_call(run_str.split(' '), cwd='docs')
        copy_tree('docs/build/html', '../atavory.github.io/ibex/')


setup(
    name='ibex',
    version='0.1.2',
    author='Ami Tavory, Shahar Azulay, Tali Raveh-Sadka',
    author_email='atavory@gmail.com',
    url='https://github.com/atavory/ibex',
    packages=[
        'ibex',
        'ibex.sklearn',
        'ibex.tensorflow',
        'ibex.tensorflow.contrib',
        'ibex.tensorflow.contrib.keras',
        'ibex.tensorflow.contrib.keras.wrappers',
        'ibex.tensorflow.contrib.keras.wrappers.scikit_learn',
        'ibex.xgboost',
    ],
    license='bsd',
    description='Pandas Adapters For Scikit-Learn',
    long_description=open('README.rst').read(),
    install_requires=['six', 'numpy', 'scipy', 'sklearn', 'pandas'],
    zip_safe=False,
    package_data={
        'ibex': ['_metadata/*txt']
    },
    include_package_data=True,
    cmdclass={
        'document': _DocumentCommand,
        'test': _TestCommand,
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
)
