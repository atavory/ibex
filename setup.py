import os
import sys
import distutils
from setuptools import setup, Command
from distutils.core import setup
import subprocess


_py2 = sys.version_info[0] < 3


class _TestCommand(Command):
    user_options = [
        ('level=', '0', 'Specify the level of the tests')
        ]

    def initialize_options(self):
        self.level = None

    def finalize_options(self):
        if self.level is None:
            self.level = '0'

    def run(self):
        run_str = "%s -m unittest discover test *test.py" % ('python' if _py2 else 'python3')
        subprocess.check_call(run_str.split(' '))


class _DocumentCommand(Command):
    user_options = [
        ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        run_str = "make html spelling"
        subprocess.check_call(run_str.split(' '), cwd='docs')


setup(
    name='ibex',
    version='0.1.0',
    author='Ami Tavory, Shahar Azulay',
    author_email='atavory at gmail',
    url='https://github.com/atavory/ibex',
    packages=[
        'ibex',
        'ibex.sklearn',
    ],
    license='bsd',
    description='Tmp Ami',
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




