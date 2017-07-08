import os
import sys
from setuptools import setup, Command
from distutils.core import setup


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
        run_str = "%s -m unittest discover frame_learn/_test '*test.py'" % ('python' if _py2 else 'python3')
       	os.system(run_str)


setup(
    name='frame_learn',
    version='0.1.0',
    author='Shahar Azulay, Tali Raveh-Sadka, Ami Tavory',
    author_email='atavory@gmail.com',
    packages=[
        'frame_learn',
    ],
    package_data={
    },
    include_package_data=True,
    license='bsd',
    description='Tmp Ami',
    long_description='Tmp Ami',
    requires=['numpy', 'pandas', 'matplotlib', 'ipython', 'pysam'],
    zip_safe=False,
    data_files=[
    ],
    cmdclass={
        'test': _TestCommand},
    classifiers=[
        'Development Status :: 5 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules'])




