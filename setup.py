from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


ext_modules=[
             Extension("cumulative",
                       sources=["cumulative.pyx"],
                       libraries=["m"] # Unix-like specific
                       )
             ]

setup(
      name = "Cumulative",
      ext_modules = cythonize(ext_modules),
      include_dirs=[numpy.get_include()]
      )