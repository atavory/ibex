import os
import sys
from setuptools import setup, Command
import subprocess


_py2 = sys.version_info[0] < 3


class _TestCommand(Command):
    user_options = [
        ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        run_str = "%s -m unittest discover test *test.py" % ('python' if _py2 else 'python3')
        cheap = os.getenv('IBEX_CHEAP_TESTS_ONLY')
        subprocess.check_call(
            run_str.split(' '),
            env={'IBEX_CHEAP_TESTS_ONLY': '1' if cheap and int(cheap) else '0'})


class _DocumentCommand(Command):
    user_options = [
        ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Tmp Ami
        # run_str = "make html spelling lint"
        run_str = "make html spelling"
        subprocess.check_call(run_str.split(' '), cwd='docs')


setup(
    name='ibex',
    version='0.1.0',
    author='Ami Tavory, Shahar Azulay, Tali Raveh-Sadka',
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
