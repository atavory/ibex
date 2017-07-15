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
        run_str = "%s -m unittest discover test '*test.py'" % ('python' if _py2 else 'python3')
       	os.system(run_str)


setup(
    name='frame_learn',
    version=open('frame_learn/_metadata/version.txt').read(),
    author=open('frame_learn/_metadata/authors.txt').read(),
    author_email='atavory@gmail.com',
    packages=[
        'frame_learn',
        'frame_learn.sklearn',
    ],
    license='bsd',
    description='Tmp Ami',
    long_description=open('README.rst').read(),
    install_requires=['six', 'numpy', 'scipy', 'sklearn', 'pandas'],
    zip_safe=False,
    package_data={
        'frame_learn': ['_metadata/*txt']
    },
    include_package_data=True,
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
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules'])




